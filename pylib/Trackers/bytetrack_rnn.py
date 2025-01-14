import os
import cv2
import lap
import sys
import json
import numpy
import torch
import imagehash
import numpy as np
from PIL import Image
from model import RNNModel
from cython_bbox import bbox_overlaps as bbox_ious
from dataloader import repr_detection, clip, read_json, read_im, get_fake_d

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
match_visual_threshold = float(sys.argv[14])

os.environ['CUDA_VISIBLE_DEVICES'] = f'{device_id}'

model_path = os.path.join(data_root, 'Trackers',
                          data_name, f'IOU_{skip_bound}.pth')

NORM = 1000.0
NUM_HIDDEN = 64

fake_d = get_fake_d(NORM)


class MoveState:
    STOP = 0
    MOVE = 1


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
        self.apparence = image
        self.appareance_feature = None

    def set_appareance_feature(self, feature):
        self.appareance_feature = feature

    def set_hidden(self, hidden):
        self.hidden = hidden

    @property
    def tlbr(self):
        return numpy.asarray([self.last_real.left,
                              self.last_real.top,
                              self.last_real.right,
                              self.last_real.bottom])


class Tracker(object):
    def __init__(self, location_model, appearance_model, transform=None):
        # for each active object, a list of detections
        self.objects = {}
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
        self.low_feature_distance_threshold = low_feature_distance_threshold

        self.max_lost_time = max_lost_time
        self.move_threshold = move_threshold

        self.keep_threshold = keep_threshold
        self.min_threshold = min_threshold
        self.create_object_threshold = create_object_threshold

    @staticmethod
    def get_visual_mat(objectfeatures, detectionfeatures, norm=True):
        visual_mat = numpy.zeros(
            (len(objectfeatures), len(detectionfeatures)), dtype='float32')
        for i, obj in enumerate(objectfeatures):
            for j, det in enumerate(detectionfeatures):
                if norm:
                    # print("norm", obj.shape, det.shape)
                    obj_normalized = obj / numpy.linalg.norm(obj)
                    det_normalized = det / numpy.linalg.norm(det)
                    visual_mat[i, j] = numpy.linalg.norm(
                        obj_normalized - det_normalized) / 2.0
        return visual_mat

    @staticmethod
    def extract_matches_based_lapjv(cost_matrix, threshold):
        if cost_matrix.size == 0:
            return numpy.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
        cost, x, y = lap.lapjv(
            cost_matrix, extend_cost=True, cost_limit=threshold)
        matches, unmatched_object, unmatched_detection = [], [], []
        for ix, mx in enumerate(x):
            if mx >= 0:
                matches.append([ix, mx])
        unmatched_object = numpy.where(x < 0)[0]
        unmatched_detection = numpy.where(y < 0)[0]
        matches = numpy.asarray(matches)
        return matches, unmatched_object, unmatched_detection

    @staticmethod
    def extract_matches_based_lapjv_with_vote(cost_matrix, threshold):
        if cost_matrix.size == 0:
            return numpy.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
        onum, dnum = cost_matrix.shape
        cost, x, y = lap.lapjv(
            cost_matrix, extend_cost=True, cost_limit=threshold)
        matches, unmatched_object, unmatched_detection = [], [], []
        voted_unmatched_object = []
        for ix, mx in enumerate(x):
            if mx == dnum - 1:
                voted_unmatched_object.append(ix)
                continue
            if mx >= 0:
                matches.append([ix, mx])
        unmatched_object = numpy.where(x < 0)[0]
        unmatched_object = voted_unmatched_object + list(unmatched_object)
        unmatched_detection = numpy.where(y < 0)[0]
        unmatched_detection = [
            di for di in unmatched_detection if di != dnum - 1]
        matches = numpy.asarray(matches)
        return matches, unmatched_object, unmatched_detection

    def get_iou_mat(self, objects, detections):
        iou_matrix = numpy.zeros(
            (len(objects), len(detections)), dtype='float32')
        object_tlbrs = numpy.ascontiguousarray(numpy.array(
            [o[1][-1].tlbr for o in objects], dtype=numpy.float64))
        detect_tlbrs = numpy.ascontiguousarray(numpy.array(
            [[d['left'], d['top'], d['right'], d['bottom']] for d in detections], dtype=numpy.float64))
        iou_matrix = 1.0 - bbox_ious(object_tlbrs, detect_tlbrs)
        return iou_matrix

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
        votes2 = {}
        for idx1, t in votes.items():
            p, idx2 = t
            if idx2 is not None and (idx2 not in votes2 or p > votes2[idx2][0]):
                votes2[idx2] = (p, idx1)
        x, y = [], []
        matches, unmatch_objects, unmatch_detections = [], [], []
        for idx2, t in votes2.items():
            p, idx1 = t
            matches.append([idx1, idx2])
            x.append(idx1)
            y.append(idx2)
        for i in range(mat.shape[0]):
            if i not in x:
                unmatch_objects.append(i)
        for j in range(mat.shape[1] - 1):
            if j not in y:
                unmatch_detections.append(j)

        # {idx1: idx2 for (idx2, (_, idx1)) in votes2.items()}
        return matches, unmatch_objects, unmatch_detections

    def get_mat(self, objects, detections, skip):
        inputs = numpy.zeros((len(objects), 10), dtype='float32')
        states = numpy.zeros((len(objects), NUM_HIDDEN), dtype='float32')
        for i, (_, obj) in enumerate(objects):
            if len(obj) >= 2:
                obj_skip = obj[-1].frame_idx - obj[-2].frame_idx
            else:
                obj_skip = 0
            inputs[i, 0:5] = repr_detection(obj_skip, obj[-1].d, NORM)
            inputs[i, 5:10] = repr_detection(
                obj[-1].frame_idx - obj[-1].last_real.frame_idx, obj[-1].last_real.d, NORM)
            states[i, :] = obj[-1].hidden
        boxes = numpy.zeros((len(detections)+1, 5), dtype='float32')
        for i, d in enumerate(detections):
            boxes[i, :] = repr_detection(skip, d, NORM)
        boxes[len(detections), :] = -1

        inputs = torch.from_numpy(inputs).cuda()
        states = torch.from_numpy(states).cuda()
        boxes = torch.from_numpy(boxes).cuda()

        with torch.no_grad():
            mat, hiddens = self.location_model.predict(inputs, states, boxes)
            mat = mat.cpu().numpy()
            hiddens = hiddens.cpu().numpy()

        return mat, hiddens

    @staticmethod
    def extract_low_features(image, hash_size=8):
        resized_img = cv2.resize(
            image, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
        gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
        mean = numpy.mean(gray_img)
        return numpy.where(gray_img > mean, 1, 0).flatten()

    @staticmethod
    def extract_low_feature_distance(lfeature1, lfeature2, hash_size):
        return numpy.sum(lfeature1 != lfeature2) / (hash_size * hash_size)

    def extract_features(self, ims):
        with torch.no_grad():
            high_dim_vectors = []
            for im in ims:
                high_dim_vector = self.appearance_model.run_on_image(im)
                high_dim_vectors.append(high_dim_vector)
            high_dim_vectors = torch.cat(high_dim_vectors, dim=0).numpy()

        return high_dim_vectors

    @staticmethod
    def compare_images(image_path1, image_path2, hash_size=8, hash_func=imagehash.average_hash):
        img1 = Image.open(image_path1)
        img2 = Image.open(image_path2)

        hash1 = hash_func(img1, hash_size=hash_size)
        hash2 = hash_func(img2, hash_size=hash_size)

        distance = hash1 - hash2

        return distance

    def update(self, frame_idx, im, detections, orig_size):
        orig_width, orig_height = orig_size
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
        # if self.appearance_model is not None:
        images = [None] * len(detections)
        # for i, d in enumerate(detections):
        #     sx = clip(d['left']*im.shape[1]//orig_width, 0, im.shape[1])
        #     sy = clip(d['top']*im.shape[0]//orig_height, 0, im.shape[0])
        #     ex = clip(d['right']*im.shape[1]//orig_width, 0, im.shape[1])
        #     ey = clip(d['bottom']*im.shape[0]//orig_height, 0, im.shape[0])
        #     if ex-sx < 4:
        #         sx = max(0, sx-2)
        #         ex = min(im.shape[1], ex+2)
        #     if ey-sy < 4:
        #         sy = max(0, sy-2)
        #         ey = min(im.shape[0], ey+2)

        #     crop = im[sy:ey, sx:ex, :]
        #     images.append(crop)

        # match each object with current frame
        objects = [(id, dlist) for id, dlist in self.objects.items() if (
            frame_idx - dlist[-1].last_real.frame_idx) < self.max_lost_time]
        object_ids = [i for i, _ in enumerate(objects)]

        keep_detections = [d for _, d in enumerate(
            detections) if d['score'] > self.keep_threshold]
        keep_detection_ids = [i for i, d in enumerate(
            detections) if d['score'] > self.keep_threshold]

        second_detections = [d for _, d in enumerate(
            detections) if d['score'] > self.min_threshold and d['score'] <= self.keep_threshold]
        second_detection_ids = [i for i, d in enumerate(
            detections) if d['score'] > self.min_threshold and d['score'] <= self.keep_threshold]

        # first match between objects and keep detections with iou
        if len(objects) > 0 and len(keep_detections) > 0:
            iou_mat = self.get_iou_mat(objects, keep_detections)
            stop_object_ids, move_object_ids, candidate_detection_ids = self.extract_matches_based_lapjv(
                iou_mat, self.move_threshold)
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
                d = Detection(keep_detections[detection_id], frame_idx, self.objects[objects[object_id]
                              [0]][-1].hidden, image=images[keep_detection_ids[detection_id]])
                self.objects[objects[object_id][0]].append(d)
                track_ids[keep_detection_ids[detection_id]
                          ] = objects[object_id][0]

            objects = [objects[oid] for oid in move_object_ids]
            object_ids = [object_ids[oid] for oid in move_object_ids]
            keep_detections = [keep_detections[did]
                               for did in candidate_detection_ids]
            keep_detection_ids = [keep_detection_ids[cdid]
                                  for cdid in candidate_detection_ids]

        # second match between objects and second detections with iou
        if len(objects) > 0 and len(second_detections) > 0:
            iou_mat = self.get_iou_mat(objects, second_detections)
            stop_object_ids, move_object_ids, candidate_detection_ids = self.extract_matches_based_lapjv(
                iou_mat, self.move_threshold)
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
                d = Detection(second_detections[detection_id], frame_idx, self.objects[objects[object_id]
                              [0]][-1].hidden, image=images[second_detection_ids[detection_id]])
                self.objects[objects[object_id][0]].append(d)
                track_ids[second_detection_ids[detection_id]
                          ] = objects[object_id][0]

            objects = [objects[oid] for oid in move_object_ids]
            object_ids = [object_ids[oid] for oid in move_object_ids]
            second_detections = [second_detections[did]
                                 for did in candidate_detection_ids]
            second_detection_ids = [second_detection_ids[cdid]
                                    for cdid in candidate_detection_ids]

        # concat_detections = keep_detections + second_detections
        # concat_detection_ids = keep_detection_ids + second_detection_ids

        # first match between objects and keep detections with mat
        # if len(objects) > 0 and len(concat_detections) > 0:
        # 	mat, hiddens = self.get_mat(objects, concat_detections, skip)
        # 	matched_object_ids, unmatched_object_ids, unmatched_detection_ids = self.extract_matches_based_vote(mat)

        # 	for object_id, detection_id in matched_object_ids:
        # 		original_detection_id = concat_detection_ids[detection_id]
        # 		d = Detection(concat_detections[detection_id], frame_idx, hiddens[object_id, :], image=images[original_detection_id])
        # 		self.objects[objects[object_id][0]].append(d)
        # 		track_ids[original_detection_id] = objects[object_id][0]

        # 	objects = [objects[oid] for oid in unmatched_object_ids]
        # 	object_ids = [object_ids[oid] for oid in unmatched_object_ids]
        # 	concat_detections = [concat_detections[did] for did in unmatched_detection_ids]
        # 	concat_detection_ids = [concat_detection_ids[did] for did in unmatched_detection_ids]

        if len(objects) > 0 and len(keep_detections) > 0:
            mat, hiddens = self.get_mat(objects, keep_detections, skip)
            matched_object_ids, unmatched_object_ids, unmatched_detection_ids = self.extract_matches_based_vote(
                mat)

            for object_id, detection_id in matched_object_ids:
                original_detection_id = keep_detection_ids[detection_id]
                d = Detection(keep_detections[detection_id], frame_idx,
                              hiddens[object_id, :], image=images[original_detection_id])
                self.objects[objects[object_id][0]].append(d)
                track_ids[original_detection_id] = objects[object_id][0]

            objects = [objects[oid] for oid in unmatched_object_ids]
            object_ids = [object_ids[oid] for oid in unmatched_object_ids]
            keep_detections = [keep_detections[did]
                               for did in unmatched_detection_ids]
            keep_detection_ids = [keep_detection_ids[did]
                                  for did in unmatched_detection_ids]

        # create new objects
        for detection_id, detection in zip(keep_detection_ids, keep_detections):
            original_detection_id = detection_id
            object_id = self.next_id
            self.next_id += 1
            d = Detection(detection, frame_idx, numpy.zeros(
                NUM_HIDDEN, dtype='float32'), image=images[original_detection_id])
            track_ids[original_detection_id] = object_id
            self.objects[object_id] = [d]

        if len(objects) > 0 and len(second_detections) > 0:
            mat, hiddens = self.get_mat(objects, second_detections, skip)
            matched_object_ids, unmatched_object_ids, unmatched_detection_ids = self.extract_matches_based_vote(
                mat)

            for object_id, detection_id in matched_object_ids:
                original_detection_id = second_detection_ids[detection_id]
                d = Detection(second_detections[detection_id], frame_idx,
                              hiddens[object_id, :], image=images[original_detection_id])
                self.objects[objects[object_id][0]].append(d)
                track_ids[original_detection_id] = objects[object_id][0]

            objects = [objects[oid] for oid in unmatched_object_ids]
            object_ids = [object_ids[oid] for oid in unmatched_object_ids]
            second_detections = [second_detections[did]
                                 for did in unmatched_detection_ids]
            second_detection_ids = [second_detection_ids[did]
                                    for did in unmatched_detection_ids]

        # create new objects
        for detection_id, detection in zip(second_detection_ids, second_detections):
            if detection['score'] < self.create_object_threshold:
                continue
            original_detection_id = detection_id
            object_id = self.next_id
            self.next_id += 1
            d = Detection(detection, frame_idx, numpy.zeros(
                NUM_HIDDEN, dtype='float32'), image=images[original_detection_id])
            track_ids[original_detection_id] = object_id
            self.objects[object_id] = [d]

        # update unmatched objects
        for object in objects:
            d = Detection(
                fake_d, frame_idx, object[1][-1].hidden, last_real=object[1][-1].last_real)
            self.objects[object[0]].append(d)

        frame_confidence = 1.0
        return track_ids, frame_confidence


if __name__ == '__main__':
    stdin = sys.stdin.detach()
    trackers = {}

    model = RNNModel().cuda()
    model.load_state_dict(torch.load(model_path), strict=False)
    model.eval()

    while True:
        packet = read_json(stdin)
        if packet is None:
            break
        id = packet['id']
        msg = packet['type']

        if msg == 'end':
            # del trackers[id]
            continue

        # im = read_im(stdin)

        detections = packet['detections']
        if detections is None:
            detections = []

        if id not in trackers:
            trackers[id] = Tracker(model, appearance_model=None)

        track_ids, frame_confidence = trackers[id].update(
            packet['frame_idx'], [], detections, packet['resolution'])
        d = {
            'outputs': track_ids,
            'conf': frame_confidence,
            't': {},
        }

        sys.stdout.write('json'+json.dumps(d)+'\n')
        sys.stdout.flush()
