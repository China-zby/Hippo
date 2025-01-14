import os
import cv2
import lap
import sys
import json
import torch
import random
import imagehash
import numpy as np
from PIL import Image
from model import RNNModel
from cython_bbox import bbox_overlaps as bbox_ious
from dataloader import repr_detection, clip, read_json, read_im, get_fake_d
from pylib.Trackers.kalman_filter import KalmanFilter

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


def _cosine_distance(a, b, data_is_normalized=False):
    if not data_is_normalized:
        a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
        b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
    return 1. - np.dot(a, b.T)


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
    def __init__(self, location_model=None):
        # for each active object, a list of detections
        self.objects = {}
        self.object_features = {}
        # processed frame indexes
        self.frames = []
        # object id counter
        self.next_id = 1
        # model
        self.location_model = KalmanFilter()

        # parameters
        self.hash_size = hash_size
        self.max_lost_time = max_lost_time
        self.move_threshold = move_threshold
        self.match_location_threshold = match_location_threshold
        self.low_feature_distance_threshold = low_feature_distance_threshold
        self.create_object_threshold = create_object_threshold

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

    @staticmethod
    def extract_matches_based_lapjv_with_vote(cost_matrix, threshold):
        if cost_matrix.size == 0:
            return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
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
        unmatched_object = np.where(x < 0)[0]
        unmatched_object = voted_unmatched_object + list(unmatched_object)
        unmatched_detection = np.where(y < 0)[0]
        unmatched_detection = [
            di for di in unmatched_detection if di != dnum - 1]
        matches = np.asarray(matches)
        return matches, unmatched_object, unmatched_detection

    def get_iou_mat(self, objects, detections):
        iou_matrix = np.zeros((len(objects), len(detections)), dtype='float32')
        object_tlbrs = np.ascontiguousarray(
            np.array([o[1][-1].tlbr for o in objects], dtype=np.float64))
        detect_tlbrs = np.ascontiguousarray(np.array(
            [[d['left'], d['top'], d['right'], d['bottom']] for d in detections], dtype=np.float64))
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
        # group by receiver and vote on max idx2->idx1 to eliminate duplicates
        votes2 = {}
        for idx1, t in votes.items():
            p, idx2 = t
            if idx2 is not None and (idx2 not in votes2 or p > votes2[idx2][0]):
                votes2[idx2] = (p, idx1)
        return {idx1: idx2 for (idx2, (_, idx1)) in votes2.items()}

    def get_mat(self, skip, objects, detections, sceneid):
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
            mat, hiddens = self.location_model.predict(
                inputs, states, boxes)
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
        ims = [self.transform(Image.fromarray(im)) for im in ims]
        ims = torch.stack(ims).cuda()
        with torch.no_grad():
            high_dim_vectors = self.appearance_model(ims).cpu().numpy()

        return high_dim_vectors

    @staticmethod
    def compare_images(image_path1, image_path2, hash_size=8, hash_func=imagehash.average_hash):
        # 读取图像
        img1 = Image.open(image_path1)
        img2 = Image.open(image_path2)

        # 计算图像哈希值
        hash1 = hash_func(img1, hash_size=hash_size)
        hash2 = hash_func(img2, hash_size=hash_size)

        # 计算哈希值之间的汉明距离
        distance = hash1 - hash2

        return distance

    def update(self, frame_idx, im, detections, orig_size, sceneid):
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
        images = [None] * len(detections)

        # match each object with current frame
        objects = [(id, dlist, frame_idx - dlist[-1].last_real.frame_idx)
                   for id, dlist in self.objects.items() if (frame_idx - dlist[-1].last_real.frame_idx) < self.max_lost_time]

        detection_ids = list(range(len(detections)))
        if len(objects) > 0 and len(detections) > 0:
            iou_mat = self.get_iou_mat(objects, detections)
            stop_object_ids, move_object_ids, candidate_detection_ids = self.extract_matches_based_lapjv(
                iou_mat, self.move_threshold)

            for object_id, detection_id in stop_object_ids:
                d = Detection(detections[detection_id], frame_idx,
                              self.objects[objects[object_id][0]][-1].hidden, image=images[detection_id])
                self.objects[objects[object_id][0]
                             ][-1].move_state = MoveState.STOP
                self.objects[objects[object_id][0]].append(d)
                track_ids[detection_id] = objects[object_id][0]

            objects = [objects[oid] for oid in move_object_ids]
            detections = [detections[did] for did in candidate_detection_ids]
            detection_ids = candidate_detection_ids

        if len(objects) > 0:
            # print('objects:', len(objects), "details:", objects)
            mat, hiddens = self.get_mat(skip, objects, detections, sceneid)
            # update objects based either on mat or gt
            matched_object_ids, unmatched_object_ids, unmatched_detection_ids = self.extract_matches_based_lapjv_with_vote(
                mat, self.match_location_threshold)

            for object_id, detection_id in matched_object_ids:
                # print(object_id, detection_id, matched_object_ids)
                original_detection_id = detection_ids[detection_id]
                d = Detection(detections[detection_id], frame_idx,
                              hiddens[object_id, :], image=images[original_detection_id])
                self.objects[objects[object_id][0]].append(d)
                track_ids[original_detection_id] = objects[object_id][0]

            objects = [objects[oid] for oid in unmatched_object_ids]
            hiddens = hiddens[unmatched_object_ids, :]
            # print(unmatched_detection_ids)
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

        frame_confidence = 1.0

        return track_ids, frame_confidence


if __name__ == '__main__':
    import cv2
    from ultralytics import YOLO
    model = YOLO("yolov8n.yaml")  # Load a model build a new model from scratch
    detector = YOLO("yolov8x.pt")
    # stdin = sys.stdin.detach()
    trackers = {}

    # model = RNNModel().cuda()
    # model.load_state_dict(torch.load(model_path), strict=False)
    # model.eval()

    skip_number = 4
    frame_idx = 1

    videoPath = "/mnt/data_ssd1/lzp/otif-dataset/dataset/hippo/test/video/1.mp4"
    videoCap = cv2.VideoCapture(videoPath)
    orig_width, orig_height = videoCap.get(
        cv2.CAP_PROP_FRAME_WIDTH), videoCap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    color_dict = {}
    track_class_list = [2]
    video_writer = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(
        *'mp4v'), max(30 // skip_number, 1), (int(orig_width), int(orig_height)))

    while True:
        ret, im = videoCap.read()

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

        print("frame_idx:", frame_idx)

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
            trackers[id] = Tracker(model)

        track_ids, frame_confidence = trackers[id].update(
            frame_idx, [], detections, [orig_width, orig_height], 0)  # , packet['resolution'], packet['sceneid'])

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
        cv2.imwrite("demo_tracker/"+str(frame_idx)+".jpg", im)

        print(d)
        frame_idx += 1
