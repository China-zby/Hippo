"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016-2020 Alex Bewley alex@bewley.ai
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import print_function

import os
import sys
import json
import random
import numpy as np

from filterpy.kalman import KalmanFilter
from dataloader import read_json

np.random.seed(0)

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
match_visual_threshold = float(sys.argv[14])
kf_pos_weight = float(sys.argv[15])
kf_vel_weight = float(sys.argv[16])

def linear_assignment(cost_matrix):
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


def iou_batch(bb_test, bb_gt):
    """
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
              + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
    return (o)


def ciou_batch(bboxes1, bboxes2):
    """
    :param bbox_p: predict of bbox(N,4)(x1,y1,x2,y2)
    :param bbox_g: groundtruth of bbox(N,4)(x1,y1,x2,y2)
    :return:
    """
    # for details should go to https://arxiv.org/pdf/1902.09630.pdf
    # ensure predict's bbox form
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    # calculate the intersection box
    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    iou_denominator = ((bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1]) +
                       (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1]) - wh + 1e-7)
    iou = wh / iou_denominator

    centerx1 = (bboxes1[..., 0] + bboxes1[..., 2]) / 2.0
    centery1 = (bboxes1[..., 1] + bboxes1[..., 3]) / 2.0
    centerx2 = (bboxes2[..., 0] + bboxes2[..., 2]) / 2.0
    centery2 = (bboxes2[..., 1] + bboxes2[..., 3]) / 2.0

    inner_diag = (centerx1 - centerx2) ** 2 + (centery1 - centery2) ** 2

    xxc1 = np.minimum(bboxes1[..., 0], bboxes2[..., 0])
    yyc1 = np.minimum(bboxes1[..., 1], bboxes2[..., 1])
    xxc2 = np.maximum(bboxes1[..., 2], bboxes2[..., 2])
    yyc2 = np.maximum(bboxes1[..., 3], bboxes2[..., 3])

    outer_diag = (xxc2 - xxc1) ** 2 + (yyc2 - yyc1) ** 2 + 1e-7

    w1 = bboxes1[..., 2] - bboxes1[..., 0]
    h1 = bboxes1[..., 3] - bboxes1[..., 1]
    w2 = bboxes2[..., 2] - bboxes2[..., 0]
    h2 = bboxes2[..., 3] - bboxes2[..., 1]

    # prevent dividing over zero. add one pixel shift
    h1 = np.maximum(h1 + 1., 1e-7)
    h2 = np.maximum(h2 + 1., 1e-7)
    arctan = np.arctan(w2/h2) - np.arctan(w1/h1)
    v = (4 / (np.pi ** 2)) * (arctan ** 2)
    S = 1 - iou
    alpha = v / (S+v)
    ciou = iou - inner_diag / outer_diag - alpha * v

    # return ciou
    return (ciou + 1) / 2.0  # resize from (-1,1) to (0,1)


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w/2.
    y = bbox[1] + h/2.
    s = w * h  # scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if (score == None):
        return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2.]).reshape((1, 4))
    else:
        return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2., score]).reshape((1, 5))


class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, bbox, frame_gap):
        """
        Initialises a tracker using initial bounding box.
        """
        # define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [
                             0, 0, 0, 1, 0, 0, 0],  [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [
                             0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        # multiply by weight
        self.kf.R[0, 0] *= kf_pos_weight
        self.kf.R[1, 1] *= kf_pos_weight
        self.kf.R[2, 2] *= kf_pos_weight
        self.kf.R[3, 3] *= kf_pos_weight
        
        self.kf.P[0, 0] *= kf_pos_weight
        self.kf.P[1, 1] *= kf_pos_weight
        self.kf.P[2, 2] *= kf_pos_weight
        self.kf.P[3, 3] *= kf_pos_weight
        
        self.kf.P[4, 4] *= kf_vel_weight
        self.kf.P[5, 5] *= kf_vel_weight
        self.kf.P[6, 6] *= kf_vel_weight

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.frame_gap = frame_gap

    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if ((self.kf.x[6]+self.kf.x[2]) <= 0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if (self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += self.frame_gap
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes)
    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if (len(trackers) == 0):
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

    iou_matrix = 1.0 - ciou_batch(detections, trackers)

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix < iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if (d not in matched_indices[:, 0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if (t not in matched_indices[:, 1]):
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if (iou_matrix[m[0], m[1]] > iou_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if (len(matches) == 0):
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort(object):
    def __init__(self, frame_gap=1, track_thresh=0.3, max_age=60, min_hits=0, iou_threshold=0.3):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        self.track_thresh = track_thresh
        self.frame_gap = frame_gap

    def update(self, output_results):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.
        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        self.frame_count += 1
        # post_process detections
        bboxes = output_results[:, :4]  # x1y1x2y2
        scores = output_results[:, 4]

        track_ids = [None] * len(bboxes)
        dets = np.concatenate(
            (bboxes, np.expand_dims(scores, axis=-1)), axis=1)
        remain_inds = scores > self.track_thresh
        remain_idxs = np.where(remain_inds)[0]
        dets = dets[remain_inds]

        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            dets, trks, self.iou_threshold)

        hit_det_ids = []
        # update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])
            hit_det_ids.append(
                [int(remain_idxs[m[0]]), self.trackers[m[1]].id])

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :], self.frame_gap)
            self.trackers.append(trk)
            hit_det_ids.append([int(remain_idxs[i]), trk.id])

        for hdi, tid in hit_det_ids:
            track_ids[hdi] = int(tid)

        i = len(self.trackers)
        for trk in reversed(self.trackers):
            i -= 1
            # remove dead tracklet
            if (trk.time_since_update > self.max_age):
                self.trackers.pop(i)

        return track_ids, 1.0


def test_tracker():
    import cv2
    from ultralytics import YOLO
    model = YOLO("yolov8n.yaml")  # Load a model build a new model from scratch
    detector = YOLO("yolov8x.pt")
    # stdin = sys.stdin.detach()
    trackers = {}

    # model = RNNModel().cuda()
    # model.load_state_dict(torch.load(model_path), strict=False)
    # model.eval()

    skip_number = 32
    frame_idx = 1

    videoPath = "/mnt/data_ssd1/lzp/otif-dataset/dataset/hippo/test/video/10.mp4"
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
            trackers[id] = Sort(frame_gap=skip_number,
                                track_thresh=keep_threshold, max_age=max_lost_time,
                                min_hits=1, iou_threshold=iou_threshold)

        detections_array = []
        for detection in detections:
            detections_array.append([detection['left'], detection['top'],
                                     detection['right'], detection['bottom'], detection['score']])
        detections_array = np.array(detections_array)

        if detections_array.size == 0:
            detections_array = np.empty((0, 5))
        # , packet['resolution'], packet['sceneid'])

        track_ids, frame_confidence = trackers[id].update(detections_array)

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
            trackers[id] = Sort(frame_gap=skip_bound,
                                track_thresh=keep_threshold, max_age=max_lost_time,
                                min_hits=1, iou_threshold=iou_threshold)

        detections_array = []
        for detection in detections:
            detections_array.append([detection['left'], detection['top'],
                                     detection['right'], detection['bottom'], detection['score']])
        detections_array = np.array(detections_array)

        if detections_array.size == 0:
            detections_array = np.empty((0, 5))

        track_ids, frame_confidence = trackers[id].update(detections_array)

        d = {
            'outputs': track_ids,
            'conf': frame_confidence,
            't': {},
        }

        sys.stdout.write('json'+json.dumps(d)+'\n')
        sys.stdout.flush()


if __name__ == '__main__':
    go_tracker()
