import os
import sys
import time
import json
import numpy
import torch
from model import RNNModel
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
kf_pos_weight = float(sys.argv[15])
kf_vel_weight = float(sys.argv[16])

os.environ['CUDA_VISIBLE_DEVICES'] = f'{device_id}'

model_path = os.path.join(data_root, 'Trackers',
                          data_name, f'IOU_{skip_bound}.pth')

NORM = 1000.0
NUM_HIDDEN = 64

fake_d = get_fake_d(NORM)


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


class Tracker(object):
    def __init__(self, model):
        # for each active object, a list of detections
        self.objects = {}
        # processed frame indexes
        self.frames = []
        # object id counter
        self.next_id = 1
        # model
        self.model = model
        self.create_object_threshold = create_object_threshold
        self.max_lost_time = max_lost_time

    def update(self, frame_idx, im, detections, resolution, gt=None):
        # cleanup frames that are now in future if needed
        if len(self.frames) > 0 and frame_idx < self.frames[-1]:
            for object_id in list(self.objects.keys()):
                self.objects[object_id] = [
                    d for d in self.objects[object_id] if d.frame_idx < frame_idx]
                if len(self.objects[object_id]) == 0:
                    del self.objects[object_id]
            self.frames = [idx for idx in self.frames if idx < frame_idx]
        self.frames.append(frame_idx)
        if len(self.frames) >= 2:
            skip = self.frames[-1] - self.frames[-2]
        else:
            skip = 0

        # get images2 if reid
        # images2 = numpy.zeros((len(detections), 64, 64, 3), dtype='uint8')
        images2 = [None]*len(detections)

        # helper func to extract matches from a matrix
        def extract_matches(mat):
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

        def get_mat(objects, detections):
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

            # mat, hiddens = session.run([m.outputs, m.out_states], feed_dict=feed_dict)
            with torch.no_grad():
                mat, hiddens = self.model.predict(inputs, states, boxes)
                mat = mat.cpu().numpy()
                hiddens = hiddens.cpu().numpy()

            return mat, hiddens  # , images2

        # match each object with current frame
        objects = [(object_id, dlist) for object_id, dlist in self.objects.items() if (
            frame_idx - dlist[-1].last_real.frame_idx) < self.max_lost_time]

        id_to_obj_idx = {}
        obj_idx_to_id = {}
        for i, (object_id, dlist) in enumerate(objects):
            id_to_obj_idx[object_id] = i
            obj_idx_to_id[i] = object_id

        if len(objects) > 0:
            # mat, hiddens, images2 = get_mat(objects, detections)
            mat, hiddens = get_mat(objects, detections)
        else:
            mat = numpy.zeros(
                (len(objects), len(detections)+1), dtype='float32')
            hiddens = numpy.zeros((len(objects), NUM_HIDDEN), dtype='float32')

        # update objects based either on mat or gt
        min_thresholds = {}
        if gt is None:
            matches = extract_matches(mat)
            track_ids = [None]*len(detections)
            for idx1 in range(len(objects)):
                if idx1 in matches:
                    idx2 = matches[idx1]
                    d = Detection(
                        detections[idx2], frame_idx, hiddens[idx1, :], image=images2[idx2])
                    track_ids[idx2] = obj_idx_to_id[idx1]
                else:
                    d = Detection(
                        fake_d, frame_idx, hiddens[idx1, :], last_real=objects[idx1][1][-1].last_real)
                objects[idx1][1].append(d)
            for idx2, d in enumerate(detections):
                if track_ids[idx2] is not None or d['score'] < self.create_object_threshold:
                    continue
                object_id = self.next_id
                self.next_id += 1
                d = Detection(d, frame_idx, numpy.zeros(
                    (NUM_HIDDEN,), dtype='float32'), image=images2[idx2])
                self.objects[object_id] = [d]
                track_ids[idx2] = object_id
        else:
            os.exit()
            # remove objects not in gt
            for object_id in list(self.objects.keys()):
                if object_id not in gt:
                    del self.objects[object_id]

            # update objects
            track_ids = [-1]*len(detections)
            for object_id, idx2 in gt.items():
                if object_id not in id_to_obj_idx:
                    self.objects[object_id] = []
                    hidden = numpy.zeros((NUM_HIDDEN,), dtype='float32')
                    last_real = None
                else:
                    idx1 = id_to_obj_idx[object_id]
                    hidden = hiddens[idx1, :]
                    last_real = self.objects[object_id][-1].last_real

                if idx2 is None:
                    d = Detection(fake_d, frame_idx, hidden,
                                  last_real=last_real)
                    if last_real is None:
                        d.image = numpy.zeros((64, 64, 3), dtype='uint8')
                else:
                    d = Detection(
                        detections[idx2], frame_idx, hidden, image=images2[idx2])
                    track_ids[idx2] = object_id
                self.objects[object_id].append(d)

                # also compute per-object min thresholds here
                if idx2 is None or object_id not in id_to_obj_idx:
                    min_thresholds[object_id] = 0.0
                    continue
                idx1 = id_to_obj_idx[object_id]
                if numpy.argmax(mat[idx1, :]) == idx2:
                    min_thresholds[object_id] = 0.0
                    continue
                min_thresholds[object_id] = float(
                    1 - (mat[idx1, idx2]+0.01)/(mat[idx1, :].max()+0.01))

        # compute frame confidence
        if mat.shape[0] > 0:
            high1 = mat.max(axis=1)
            mat_ = numpy.copy(mat)
            mat_[numpy.arange(mat_.shape[0]), numpy.argmax(mat_, axis=1)] = 0
            high2 = mat_.max(axis=1)
            track_confidences = 1 - (high2+0.01)/(high1+0.01)
            frame_confidence = float(numpy.min(track_confidences))
        else:
            frame_confidence = 1.0

        # pickle.dump([track_ids, frame_confidence, min_thresholds], open("tracker_result.pkl", "wb"))
        return track_ids, frame_confidence, min_thresholds


if __name__ == '__main__':
    stdin = sys.stdin.detach()
    trackers = {}

    model = RNNModel().cuda()
    model.load_state_dict(torch.load(model_path), strict=False)
    model.eval()

    while True:
        t0 = time.time()
        packet = read_json(stdin)
        if packet is None:
            break
        vid = packet['id']
        msg = packet['type']
        frame_idx = packet['frame_idx']

        if msg == 'end':
            continue

        # im = read_im(stdin)

        detections = packet['detections']
        if detections is None:
            detections = []

        if vid not in trackers:
            trackers[vid] = Tracker(model)

        track_ids, frame_confidence, min_thresholds = trackers[vid].update(
            frame_idx, [], detections, packet['resolution'])
        t0 = time.time()
        d = {
            'outputs': track_ids,
            'conf': frame_confidence,
            't': min_thresholds,
        }
        sys.stdout.write('json'+json.dumps(d)+'\n')
        sys.stdout.flush()
