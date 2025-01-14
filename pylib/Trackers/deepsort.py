import os
import sys
import cv2
import json
import torch
import numpy as np

import iou_matching
import kalman_filter
from track import Track
import linear_assignment
from detection import Detection
from reid_model import Extractor
from dataloader import read_json, read_im

sys.path.append("./pylib/Reid/")
from fastreid.config import get_cfg
from demo.predictor import FeatureExtractionDemo

data_root = sys.argv[1]
data_name = sys.argv[2]
skip_bound = int(sys.argv[3])
device_id = sys.argv[4]

hash_size = int(sys.argv[5])
low_feature_distance_threshold = float(sys.argv[6])
max_lost_time = int(sys.argv[7])
move_threshold = float(sys.argv[8])
keep_threshold = float(sys.argv[9])
min_threshold = float(sys.argv[10])
create_object_threshold = float(sys.argv[11])
iou_threshold = float(sys.argv[12])
unmatch_location_threshold = float(sys.argv[13])
visual_threshold = float(sys.argv[14])
kf_pos_weight = float(sys.argv[15])
kf_vel_weight = float(sys.argv[16])

def setup_cfg(config_file, weight_file):
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = weight_file
    cfg.freeze()
    return cfg


def _cosine_distance(a, b, data_is_normalized=False):
    if not data_is_normalized:
        a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
        b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
    return 1. - np.dot(a, b.T)


def _nn_cosine_distance(x, y):
    distances = _cosine_distance(x, y)
    return distances.min(axis=0)


class Tracker:
    def __init__(self, metric, max_iou_distance=0.1, max_age=70, n_init=3, frame_gap=1):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

        self.kf = kalman_filter.KalmanFilter(kf_pos_weight, kf_vel_weight)
        self.tracks = []
        self._next_id = 1
        self.frame_gap = frame_gap

    def predict(self):
        """Propagate track state distributions one time step forward.
        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.kf)

    # def increment_ages(self):
    #     for track in self.tracks:
    #         track.increment_age()
    #         track.mark_missed()

    def update(self, detections, classes, remain_idxs, hit_det_ids):
        """Perform measurement update and track management.
        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.
        """
        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)

        # Update track set.
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(
                self.kf, detections[detection_idx])
            hit_det_ids.append([remain_idxs[detection_idx],
                               self.tracks[track_idx].track_id])
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            self._initiate_track(
                detections[detection_idx], classes[detection_idx].item())
            hit_det_ids.append(
                [remain_idxs[detection_idx], self.tracks[-1].track_id])
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)
        return hit_det_ids

    def _match(self, detections):
        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices)

            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # Associate confirmed tracks using appearance features.
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed_tracks)

        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]
        
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.ciou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection, class_id):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(Track(
            mean, covariance, self._next_id, class_id, self.n_init, self.max_age,
            detection.feature, self.frame_gap))
        self._next_id += 1


class NearestNeighborDistanceMetric(object):
    def __init__(self, metric, matching_threshold, budget=None):

        if metric == "cosine":
            self._metric = _nn_cosine_distance
        else:
            raise ValueError(
                "Invalid metric; must be either 'euclidean' or 'cosine'")
        self.matching_threshold = matching_threshold
        self.budget = budget
        self.samples = {}

    def partial_fit(self, features, targets, active_targets):
        for feature, target in zip(features, targets):
            self.samples.setdefault(target, []).append(feature)
            if self.budget is not None:
                self.samples[target] = self.samples[target][-self.budget:]
        self.samples = {k: self.samples[k] for k in active_targets}

    def distance(self, features, targets):
        cost_matrix = np.zeros((len(targets), len(features)))
        for i, target in enumerate(targets):
            cost_matrix[i, :] = self._metric(self.samples[target], features)
        return cost_matrix


class DeepSort(object):
    def __init__(self, cfg_path=None, weight_path=None,
                 max_dist=0.1, track_thresh=0.3,
                 iou_threshold=0.3,
                 max_age=60, n_init=1, nn_budget=100, frame_gap=1):
        self.track_thresh = track_thresh

        cfg = setup_cfg(cfg_path, weight_path)
        self.extractor = FeatureExtractionDemo(cfg)

        max_cosine_distance = max_dist
        metric = NearestNeighborDistanceMetric(
            "cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(
            metric, max_iou_distance=iou_threshold, max_age=max_age, n_init=n_init, frame_gap=frame_gap)

    def update(self, output_results, ori_img, orishape):
        self.height, self.width = ori_img.shape[:2]

        # post process detections
        confidences = output_results[:, 4]
        bboxes = output_results[:, :4]  # x1y1x2y2
        track_ids = [None] * len(bboxes)

        bbox_xyxy = bboxes
        bbox_tlwh = self._xyxy_to_tlwh_array(bbox_xyxy)
        remain_inds = confidences > self.track_thresh
        bbox_tlwh = bbox_tlwh[remain_inds]
        confidences = confidences[remain_inds]

        remain_idxs = np.where(remain_inds)[0]
        hit_det_ids = []

        # generate detections
        features = self._get_features(bbox_tlwh, ori_img, orishape)
        detections = [Detection(bbox_tlwh[i], conf, features[i]) for i, conf in enumerate(
            confidences) if conf > self.track_thresh]
        classes = np.zeros((len(detections), ))

        # update tracker
        self.tracker.predict()
        hit_det_ids = self.tracker.update(
            detections, classes, remain_idxs, hit_det_ids)

        # output bbox identities
        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            box = track.to_tlwh()
            x1, y1, x2, y2 = self._tlwh_to_xyxy_noclip(box)
            track_id = track.track_id
            class_id = track.class_id
            outputs.append(
                np.array([x1, y1, x2, y2, track_id, class_id], dtype=np.int))
        if len(outputs) > 0:
            outputs = np.stack(outputs, axis=0)

        for hdi, tid in hit_det_ids:
            track_ids[hdi] = int(tid)
        return track_ids, 1.0

    """
    TODO:
        Convert bbox from xc_yc_w_h to xtl_ytl_w_h
    Thanks JieChen91@github.com for reporting this bug!
    """
    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone()
        bbox_tlwh[:, 0] = bbox_xywh[:, 0] - bbox_xywh[:, 2] / 2.
        bbox_tlwh[:, 1] = bbox_xywh[:, 1] - bbox_xywh[:, 3] / 2.
        return bbox_tlwh

    @staticmethod
    def _xyxy_to_tlwh_array(bbox_xyxy):
        if isinstance(bbox_xyxy, np.ndarray):
            bbox_tlwh = bbox_xyxy.copy()
        elif isinstance(bbox_xyxy, torch.Tensor):
            bbox_tlwh = bbox_xyxy.clone()
        bbox_tlwh[:, 2] = bbox_xyxy[:, 2] - bbox_xyxy[:, 0]
        bbox_tlwh[:, 3] = bbox_xyxy[:, 3] - bbox_xyxy[:, 1]
        return bbox_tlwh

    def _xywh_to_xyxy(self, bbox_xywh):
        x, y, w, h = bbox_xywh
        x1 = max(int(x - w / 2), 0)
        x2 = min(int(x + w / 2), self.width - 1)
        y1 = max(int(y - h / 2), 0)
        y2 = min(int(y + h / 2), self.height - 1)
        return x1, y1, x2, y2

    def _tlwh_to_xyxy(self, bbox_tlwh, imshape, orishape):
        """
        TODO:
            Convert bbox from xtl_ytl_w_h to xc_yc_w_h
        Thanks JieChen91@github.com for reporting this bug!
        """
        x, y, w, h = bbox_tlwh
        x1 = max(int(x * imshape[1] // orishape[0]), 0)
        x2 = min(int((x+w) * imshape[1] // orishape[0]), self.width - 1)
        y1 = max(int(y * imshape[0] // orishape[1]), 0)
        y2 = min(int((y+h) * imshape[0] // orishape[1]), self.height - 1)
        return x1, y1, x2, y2

    def _tlwh_to_xyxy_noclip(self, bbox_tlwh):
        """
        TODO:
            Convert bbox from xtl_ytl_w_h to xc_yc_w_h
        Thanks JieChen91@github.com for reporting this bug!
        """
        x, y, w, h = bbox_tlwh
        x1 = x
        x2 = x + w
        y1 = y
        y2 = y + h
        return x1, y1, x2, y2

    # def increment_ages(self):
    #     self.tracker.increment_ages()

    def _xyxy_to_tlwh(self, bbox_xyxy):
        x1, y1, x2, y2 = bbox_xyxy

        t = x1
        l = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        return t, l, w, h

    def _get_features(self, bbox_xywh, ori_img, orishape):
        im_crops = []
        for box in bbox_xywh:
            x1, y1, x2, y2 = self._tlwh_to_xyxy(box, ori_img.shape, orishape)
            if x1 >= x2 or y1 >= y2:
                x2 = x1 + 10
                y2 = y1 + 10
            im = ori_img[y1:y2, x1:x2]
            im_crops.append(im)
        if im_crops:
            features = self.extractor(im_crops)
        else:
            features = np.array([])
        return features


def test_tracker():
    import cv2
    import random
    from ultralytics import YOLO
    model = YOLO("yolov8x.yaml")  # Load a model build a new model from scratch
    detector = YOLO("yolov8x.pt")
    # stdin = sys.stdin.detach()
    trackers = {}

    # model = RNNModel().cuda()
    # model.load_state_dict(torch.load(model_path), strict=False)
    # model.eval()
    cfg_path = f"./pylib/Reid/configs/Videodb/camera_envs.yml"
    weight_path = f"./pylib/Reid/logs/videodb/camera_envs/model_best.pth"

    skip_number = 32
    frame_idx = 1

    videoPath = "/mnt/data_ssd1/lzp/otif-dataset/dataset/hippo/test/video/3.mp4"
    videoCap = cv2.VideoCapture(videoPath)
    orig_width, orig_height = videoCap.get(
        cv2.CAP_PROP_FRAME_WIDTH), videoCap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    color_dict = {}
    track_class_list = [2]
    video_writer = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(
        *'mp4v'), max(30 // skip_number, 1), (int(orig_width), int(orig_height)))

    while True:
        ret, im = videoCap.read()
        if not ret:
            break

        # if not ret:
        #     break
        # packet = read_json(stdin)
        # if packet is None:
        #     break
        # id = packet['id']
        id = 0
        # msg = packet['type']
        # frame_idx = packet['frame_idx']

        # if msg == 'end':
        #     # del trackers[id]
        #     continue

        if frame_idx % skip_number != 0:
            frame_idx += 1
            continue

        # im = read_im(stdin)
        # im = im[:, :, ::-1]
        results = detector(im, verbose=False, iou=0.45)
        detections = []
        for result in results:
            # print(result.boxes[0], result.boxes.shape)
            xyxys = result.boxes.xyxy
            confs = result.boxes.conf
            clses = result.boxes.cls
            for xyxy, conf, cls in zip(xyxys, confs, clses):
                classID = int(cls)
                # print(classID)
                if classID not in track_class_list:
                    continue
                detection = {"left": int(xyxy[0]), "top": int(xyxy[1]), "right": int(
                    xyxy[2]), "bottom": int(xyxy[3]), "score": float(conf), "class": int(cls)}
                detections.append(detection)

        # detections = packet['detections']
        # print("detections", detections, file=open(f"./demos/log+{id}.txt", "a"))
        if detections is None:
            detections = []

        if id not in trackers:
            trackers[id] = DeepSort(cfg_path, weight_path,
                                    max_dist=visual_threshold, track_thresh=keep_threshold,
                                    iou_threshold=iou_threshold,
                                    max_age=max_lost_time, n_init=1, nn_budget=100, frame_gap=skip_number)

        detections_array = []
        for detection in detections:
            detections_array.append([detection['left'], detection['top'],
                                     detection['right'], detection['bottom'], detection['score']])
        detections_array = np.array(detections_array)

        if detections_array.size == 0:
            detections_array = np.empty((0, 5))
        # , packet['resolution'], packet['sceneid'])

        ori_shape = [im.shape[1], im.shape[0]]
        track_ids, frame_confidence = trackers[id].update(detections_array, im, ori_shape)

        # print("track_ids", track_ids, file=open(f"./demos/log+{id}.txt", "a"))
        d = {
            'outputs': track_ids,
            'conf': frame_confidence,
            't': {},
        }
        # print("d", d, file=open(f"./demos/log+{id}.txt", "a"))
        # sys.stdout.write('json'+json.dumps(d)+'\n')
        # sys.stdout.flush()

        for detection, track_id in zip(detections, track_ids):  # visualize
            if track_id is None:
                continue
            if track_id not in color_dict:
                color_dict[track_id] = (random.randint(
                    0, 255), random.randint(0, 255), random.randint(0, 255))
            cv2.rectangle(im, (detection['left'], detection['top']), (
                detection['right'], detection['bottom']), color_dict[track_id], 2)
            cv2.putText(im, str(track_id), (detection['left'], detection['top']),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color_dict[track_id], 2)
        video_writer.write(im)
        cv2.imwrite("./demo_tracker/"+str(frame_idx)+".jpg", im)

        # print(d)
        frame_idx += 1


def go_tracker():
    stdin = sys.stdin.detach()
    trackers = {}

    cfg_path = f"./pylib/Reid/configs/Videodb/{data_name}.yml"
    weight_path = f"./pylib/Reid/logs/videodb/{data_name}/model_best.pth"

    while True:
        packet = read_json(stdin)
        if packet is None:
            break
        id = packet['id']
        msg = packet['type']
        frame_idx = packet['frame_idx']

        if msg == 'end':
            # del trackers[id]
            continue

        detections = packet['detections']
        if detections is None:
            detections = []

        if id not in trackers:
            trackers[id] = DeepSort(cfg_path, weight_path,
                                    max_dist=visual_threshold, track_thresh=keep_threshold,
                                    iou_threshold=iou_threshold,
                                    max_age=max_lost_time, n_init=1, nn_budget=100, frame_gap=skip_bound)

        detections_array = []
        for detection in detections:
            detections_array.append([detection['left'], detection['top'],
                                     detection['right'], detection['bottom'], detection['score']])
        detections_array = np.array(detections_array)

        if detections_array.size == 0:
            detections_array = np.empty((0, 5))

        im = read_im(stdin)
        im = im[:, :, ::-1]

        track_ids, frame_confidence = trackers[id].update(
            detections_array, im, packet['resolution'])

        d = {
            'outputs': track_ids,
            'conf': frame_confidence,
            't': {},
        }

        sys.stdout.write('json'+json.dumps(d)+'\n')
        sys.stdout.flush()


if __name__ == '__main__':
    # test_tracker()
    go_tracker()
