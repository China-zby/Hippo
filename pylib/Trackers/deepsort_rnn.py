import os
import cv2
import lap
import sys
import json
import torch
import random
import numpy as np
from model import RNNModel
from cython_bbox import bbox_overlaps as bbox_ious
from scipy.optimize import linear_sum_assignment as linear_assignment
from dataloader import repr_detection, clip, read_json, read_im, get_fake_d

sys.path.append("./pylib/Reid/")
from fastreid.config import get_cfg
from demo.predictor import FeatureExtractionDemo

def setup_cfg(config_file, weight_file):
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = weight_file
    cfg.freeze()
    return cfg


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
match_location_threshold = float(sys.argv[12])
unmatch_location_threshold = float(sys.argv[13])
visual_threshold = float(sys.argv[14])

os.environ['CUDA_VISIBLE_DEVICES'] = f'{device_id}'

model_path = os.path.join(data_root, 'Trackers',
                          data_name, f'IOU_{skip_bound}.pth')

NORM = 1000.0
NUM_HIDDEN = 64

fake_d = get_fake_d(NORM)


class MoveState:
    STOP = 0
    MOVE = 1


def _cosine_distance(a, b, data_is_normalized=False):
    if not data_is_normalized:
        a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
        b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
    return 1. - np.dot(a, b.T)


def _nn_cosine_distance(x, y):
    distances = _cosine_distance(x, y)
    return distances.min(axis=0)


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


def min_cost_matching(
        distance_metric, max_distance, tracks, detections, location_mat, track_indices=None,
        detection_indices=None):
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    if len(detection_indices) == 0 or len(track_indices) == 0:
        return [], track_indices, detection_indices  # Nothing to match.

    cost_matrix = distance_metric(
        tracks, detections, location_mat, track_indices, detection_indices)
    cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5

    row_indices, col_indices = linear_assignment(cost_matrix)

    matches, unmatched_tracks, unmatched_detections = [], [], []
    for col, detection_idx in enumerate(detection_indices):
        if col not in col_indices:
            unmatched_detections.append(detection_idx)

    for row, track_idx in enumerate(track_indices):
        if row not in row_indices:
            unmatched_tracks.append(track_idx)

    for row, col in zip(row_indices, col_indices):
        track_idx = track_indices[row]
        detection_idx = detection_indices[col]
        if cost_matrix[row, col] > max_distance:
            unmatched_tracks.append(track_idx)
            unmatched_detections.append(detection_idx)
        else:
            matches.append((track_idx, detection_idx))

    return matches, unmatched_tracks, unmatched_detections


class Detection(object):
    def __init__(self, d, frame_idx, hidden, last_real=None, image=None):
        self.d = d
        self.frame_idx = frame_idx
        self.hidden = hidden
        self.left = d['left']
        self.top = d['top']
        self.right = d['right']
        self.bottom = d['bottom']

        self.image = image
        if last_real is None:
            self.last_real = self
        else:
            self.last_real = last_real

        # for tracking
        self.move_state = MoveState.MOVE
        self.apparence = image
        self.appareance_feature = None

    def set_appareance_feature(self, feature):
        self.appareance_feature = feature

    def set_hidden(self, hidden):
        self.hidden = hidden

    @property
    def tlbr(self):
        return np.asarray([self.last_real.left,
                           self.last_real.top,
                           self.last_real.right,
                           self.last_real.bottom])


class Tracker(object):
    def __init__(self, location_model, appearance_model, transform=None):
        # for each active object, a list of detections
        self.objects = {}
        self.object_features = {}
        # processed frame indexes
        self.frames = []
        # object id counter
        self.next_id = 1
        # model
        self.location_model = location_model
        self.appearance_model = appearance_model
        self.transform = transform

        # parameters
        self.hash_size = hash_size
        self.max_lost_time = max_lost_time

        self.move_threshold = move_threshold
        self.match_location_threshold = match_location_threshold
        self.unmatch_location_threshold = unmatch_location_threshold
        self.visual_threshold = visual_threshold
        self.low_feature_distance_threshold = low_feature_distance_threshold
        self.create_object_threshold = create_object_threshold
        self.metric = _nn_cosine_distance

    @staticmethod
    def get_visual_mat(objectfeatures, detectionfeatures, norm=True):
        visual_mat = np.zeros(
            (len(objectfeatures), len(detectionfeatures)), dtype='float32')
        for i, obj in enumerate(objectfeatures):
            for j, det in enumerate(detectionfeatures):
                if norm:
                    # print("norm", obj.shape, det.shape)
                    obj_normalized = obj / np.linalg.norm(obj)
                    det_normalized = det / np.linalg.norm(det)
                    visual_mat[i, j] = np.linalg.norm(
                        obj_normalized - det_normalized) / 2.0
        return visual_mat

    def get_iou_mat(self, objects, detections):
        iou_matrix = np.zeros((len(objects), len(detections)), dtype='float32')
        object_tlbrs = np.ascontiguousarray(
            np.array([o[1][-1].tlbr for o in objects], dtype=np.float64))
        detect_tlbrs = np.ascontiguousarray(np.array(
            [[d['left'], d['top'], d['right'], d['bottom']] for d in detections], dtype=np.float64))
        iou_matrix = 1.0 - bbox_ious(object_tlbrs, detect_tlbrs)
        return iou_matrix

    @staticmethod
    def extract_matches_based_lapjv(cost_matrix, threshold):
        if cost_matrix.size == 0:
            return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
        if cost_matrix.shape[0] == 1 and cost_matrix.shape[1] == 1:
            # print(cost_matrix[0, 0], "assaasasasassa")
            if cost_matrix[0, 0] < threshold:
                return np.asarray([[0, 0]]), (), ()
            else:
                # print(cost_matrix[0, 0], "assaasasasassaascccccqqwqwqwqwqwqwqwqwqwqwqwqw", np.empty((0, 2), dtype=int), (0,), (0,))
                return np.empty((0, 2), dtype=int), (0,), (0,)
        cost, x, y = lap.lapjv(
            cost_matrix, extend_cost=True, cost_limit=threshold)
        matches, unmatched_object, unmatched_detection = [], [], []
        for ix, mx in enumerate(x):
            if mx >= 0:
                matches.append([ix, mx])
        unmatched_object = np.where(x < 0)[0]
        unmatched_detection = np.where(y < 0)[0]
        matches = np.asarray(matches)
        return matches, unmatched_object, unmatched_detection

    # helper func to extract matches from a matrixchat
    @staticmethod
    def extract_matches_based_vote(mat):
        votes = {}
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                p = mat[i, j]
                if i not in votes or p > votes[i][0]:
                    if j == mat.shape[1]-1:
                        votes[i] = (p, None)
                    else:
                        votes[i] = (p, j)
        # group by receiver and vote on max idx2->idx1 to eliminate duplicates
        votes2 = {}
        for idx1, t in votes.items():
            p, idx2 = t
            if idx2 is not None and (idx2 not in votes2 or p > votes2[idx2][0]):
                votes2[idx2] = (p, idx1)
        return {idx1: idx2 for (idx2, (_, idx1)) in votes2.items()}

    def get_mat(self, skip, objects, detections):
        inputs = np.zeros((len(objects), 10), dtype='float32')
        states = np.zeros((len(objects), NUM_HIDDEN), dtype='float32')
        for i, (_, obj, _) in enumerate(objects):
            if len(obj) >= 2:
                obj_skip = obj[-1].frame_idx - obj[-2].frame_idx
            else:
                obj_skip = 0
            inputs[i, 0:5] = repr_detection(obj_skip, obj[-1].d, NORM)
            inputs[i, 5:10] = repr_detection(
                obj[-1].frame_idx - obj[-1].last_real.frame_idx, obj[-1].last_real.d, NORM)
            states[i, :] = obj[-1].hidden
        boxes = np.zeros((len(detections)+1, 5), dtype='float32')
        for i, d in enumerate(detections):
            boxes[i, :] = repr_detection(skip, d, NORM)
        boxes[len(detections), :] = -1

        inputs = torch.from_numpy(inputs).cuda()
        states = torch.from_numpy(states).cuda()
        boxes = torch.from_numpy(boxes).cuda()

        with torch.no_grad():
            mat, hiddens = self.location_model.predict(inputs, states, boxes)
            mat = 1.0 - mat.cpu().numpy()
            hiddens = hiddens.cpu().numpy()

        return mat, hiddens

    @staticmethod
    def extract_low_features(image, hash_size=8):
        resized_img = cv2.resize(
            image, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
        gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
        mean = np.mean(gray_img)
        return np.where(gray_img > mean, 1, 0).flatten()

    @staticmethod
    def extract_low_feature_distance(lfeature1, lfeature2, hash_size):
        return np.sum(lfeature1 != lfeature2) / (hash_size * hash_size)

    def extract_features(self, ims):
        with torch.no_grad():
            high_dim_vectors = []
            for im in ims:
                high_dim_vector = self.appearance_model.run_on_image(im)
                high_dim_vectors.append(high_dim_vector)
            high_dim_vectors = torch.cat(high_dim_vectors, dim=0).numpy()

        return high_dim_vectors

    def gated_metric(self, objects, detections, mat, objectIndexs, detectionIndexs, gated_cost=1e+5):
        objects = [objects[i] for i in objectIndexs]
        detections = [detections[i] for i in detectionIndexs]
        mat = mat[objectIndexs, :][:, detectionIndexs]
        detectionNum = len(detections) - 1
        features = [detection['feature'] for detection in detections]

        cost_matrix = np.zeros((len(objects), len(features)))
        for i, objectIns in enumerate(objects):
            cost_matrix[i, :] = self.metric(
                self.object_features[objectIns[0]], features)
            if mat[i, detectionNum] < self.unmatch_location_threshold:
                cost_matrix[i, :] = gated_cost
            else:
                for j in range(len(features)):
                    if mat[i, j] > self.match_location_threshold:
                        cost_matrix[i, j] = gated_cost

        return cost_matrix

    def extract_matches_based_cascade(self,
                                      distance_metric,
                                      cascade_depth,
                                      objects,
                                      detections, location_mat,
                                      track_indices=None, detection_indices=None):
        if track_indices is None:
            track_indices = list(range(len(objects)))
        if detection_indices is None:
            detection_indices = list(range(len(detections)))

        unmatched_detections = detection_indices
        matches = []
        for level in range(cascade_depth):
            if len(unmatched_detections) == 0:  # No detections left
                break

            track_indices_l = [
                k for k in track_indices
                if objects[k][2] == level
            ]
            if len(track_indices_l) == 0:  # Nothing to match at this level
                continue

            matches_l, _, unmatched_detections = \
                min_cost_matching(
                    distance_metric, self.visual_threshold, objects, detections, location_mat,
                    track_indices_l, unmatched_detections)
            matches += matches_l
        unmatched_tracks = list(set(track_indices) -
                                set(k for k, _ in matches))
        return matches, unmatched_tracks, unmatched_detections

    def update(self, frame_idx, im, detections, orishape):
        orig_width, orig_height = orishape
        track_ids = [None]*len(detections)

        # cleanup frames that are now in future if needed
        if len(self.frames) > 0 and frame_idx < self.frames[-1]:
            for id in list(self.objects.keys()):
                self.objects[id] = [d for d in self.objects[id]
                                    if d.frame_idx < frame_idx]
                if len(self.objects[id]) == 0:
                    del self.objects[id]
            self.frames = [idx for idx in self.frames if idx < frame_idx]
        self.frames.append(frame_idx)

        if len(self.frames) >= 2:
            skip = self.frames[-1] - self.frames[-2]
        else:
            skip = 0

        # get images if utilize
        if self.appearance_model is not None:
            images = []
            for i, d in enumerate(detections):
                sx = clip(d['left']*im.shape[1]//orig_width, 0, im.shape[1])
                sy = clip(d['top']*im.shape[0]//orig_height, 0, im.shape[0])
                ex = clip(d['right']*im.shape[1]//orig_width, 0, im.shape[1])
                ey = clip(d['bottom']*im.shape[0]//orig_height, 0, im.shape[0])
                if ex-sx < 4:
                    sx = max(0, sx-2)
                    ex = min(im.shape[1], ex+2)
                if ey-sy < 4:
                    sy = max(0, sy-2)
                    ey = min(im.shape[0], ey+2)

                crop = im[sy:ey, sx:ex, :]
                images.append(crop)
            if len(images) > 0:
                features = self.extract_features(images)
                for di, feature in enumerate(features):
                    detections[di]['feature'] = feature

        # match each object with current frame
        objects = [(id, dlist, frame_idx - dlist[-1].last_real.frame_idx)
                   for id, dlist in self.objects.items() if (frame_idx - dlist[-1].last_real.frame_idx) < self.max_lost_time]
        detection_ids = list(range(len(detections)))

        if len(objects) > 0 and len(detections) > 0:
            iou_mat = self.get_iou_mat(objects, detections)
            # print(iou_mat.shape, len(objects), len(detections))
            stop_object_ids, move_object_ids, candidate_detection_ids = self.extract_matches_based_lapjv(
                iou_mat, self.move_threshold)
            # print('stop_object_ids', stop_object_ids)

            for object_id, detection_id in stop_object_ids:
                # object_low_feature, detection_low_feature = self.extract_low_features(
                #     objects[object_id][1][-1].last_real.image, self.hash_size), self.extract_low_features(images[detection_id], self.hash_size)
                # low_feature_distance = self.extract_low_feature_distance(
                #     object_low_feature, detection_low_feature, self.hash_size)
                # if low_feature_distance > self.low_feature_distance_threshold:
                #     move_object_ids = np.array(
                #         list(move_object_ids) + [object_id])
                #     candidate_detection_ids = np.array(
                #         list(candidate_detection_ids) + [detection_id])
                #     continue
                d = Detection(detections[detection_id], frame_idx,
                              self.objects[objects[object_id][0]][-1].hidden, image=images[detection_id])
                self.objects[objects[object_id][0]
                             ][-1].move_state = MoveState.STOP
                self.objects[objects[object_id][0]].append(d)
                self.object_features[objects[object_id]
                                     [0]].append(features[detection_id])
                track_ids[detection_id] = objects[object_id][0]

            objects = [objects[oid] for oid in move_object_ids]
            detections = [detections[did] for did in candidate_detection_ids]
            detection_ids = [detection_ids[cdid]
                             for cdid in candidate_detection_ids]

        if len(objects) > 0:
            mat, hiddens = self.get_mat(skip, objects, detections)
            matched_object_ids, unmatched_object_ids, unmatched_detection_ids = self.extract_matches_based_cascade(self.gated_metric,
                                                                                                                   self.max_lost_time,
                                                                                                                   objects,
                                                                                                                   detections,
                                                                                                                   mat)

            for object_id, detection_id in matched_object_ids:
                original_detection_id = detection_ids[detection_id]
                d = Detection(detections[detection_id], frame_idx,
                              hiddens[object_id, :], image=images[original_detection_id])
                self.objects[objects[object_id][0]].append(d)
                self.object_features[objects[object_id][0]].append(
                    features[original_detection_id])
                track_ids[original_detection_id] = objects[object_id][0]

            objects = [objects[oid] for oid in unmatched_object_ids]
            hiddens = hiddens[unmatched_object_ids, :]
            detections = [detections[did] for did in unmatched_detection_ids]
            detection_ids = [detection_ids[did]
                             for did in unmatched_detection_ids]
        else:
            mat = np.zeros((0, len(detections)), dtype='float32')
            hiddens = np.zeros((0, NUM_HIDDEN), dtype='float32')

        # update unmatched objects
        for objectIns in objects:
            d = Detection(
                fake_d, frame_idx, objectIns[1][-1].hidden, last_real=objectIns[1][-1].last_real)
            self.objects[objectIns[0]].append(d)

        # create new objects
        for detection_id, detection in zip(detection_ids, detections):
            # print(detection_id, detection)
            if detection['score'] < self.create_object_threshold:
                continue
            original_detection_id = detection_id
            object_id = self.next_id
            self.next_id += 1
            d = Detection(detection, frame_idx, np.zeros(
                NUM_HIDDEN, dtype='float32'), image=images[original_detection_id])
            track_ids[original_detection_id] = object_id
            self.objects[object_id] = [d]
            self.object_features[object_id] = [features[original_detection_id]]

        frame_confidence = 1.0

        return track_ids, frame_confidence


if __name__ == '__main__':
    stdin = sys.stdin.detach()
    trackers = {}

    model = RNNModel().cuda()
    model.load_state_dict(torch.load(model_path), strict=False)
    model.eval()

    cfg_path = f"./pylib/Reid/configs/Videodb/{data_name}.yml"
    weight_path = f"./pylib/Reid/logs/videodb/{data_name}/model_best.pth"
    cfg = setup_cfg(cfg_path, weight_path)
    appearance_model = FeatureExtractionDemo(cfg)

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

        im = read_im(stdin)
        im = im[:, :, ::-1]

        detections = packet['detections']
        if detections is None:
            detections = []

        if id not in trackers:
            trackers[id] = Tracker(model, appearance_model)

        track_ids, frame_confidence = trackers[id].update(
            frame_idx, im, detections, packet['resolution'])

        d = {
            'outputs': track_ids,
            'conf': frame_confidence,
            't': {},
        }

        sys.stdout.write('json'+json.dumps(d)+'\n')
        sys.stdout.flush()
