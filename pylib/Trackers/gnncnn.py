import os
import cv2
import lap
import sys
import math
import json
import torch
import imagehash
import numpy as np
from PIL import Image
from dataloader import get_loc
from model import MOTGraphModel
from torchvision import transforms
from torch_geometric.data import Data
from cython_bbox import bbox_overlaps as bbox_ious
from dataloader import clip, read_json, read_im, get_fake_d

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

model_path = os.path.join(data_root, 'Trackers', data_name, f'GNNCNN_{skip_bound}.pth')

NORM = 1000.0
SIZE_CROP = 64

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

class Detection(object):
    def __init__(self, d, frame_idx, last_real=None, image=None):
        self.d = d
        self.frame_idx = frame_idx
        self.left = d['left']
        self.top = d['top']
        self.right = d['right']
        self.bottom = d['bottom']

        self.image = image
        if last_real is None: self.last_real = self
        else: self.last_real = last_real
   
        # for tracking
        self.move_state = MoveState.MOVE
        self.apparence = image
        self.appareance_feature = None
  
    def set_appareance_feature(self, feature):
        self.appareance_feature = feature
   
    @property
    def tlbr(self):
        return np.asarray([self.last_real.left, 
                              self.last_real.top, 
                              self.last_real.right, 
                              self.last_real.bottom])

class Tracker(object):
    def __init__(self, gnncnn_model, transform, device):
        # for each active object, a list of detections
        self.objects = {}
        # self.object_crops = {}
        # processed frame indexes
        self.frames = []
        # object id counter
        self.next_id = 1
        # model
        self.device = device
        self.gnncnn_model = gnncnn_model
        self.transform = transform
    
        # parameters
        self.hash_size = hash_size
        self.max_lost_time = max_lost_time
        self.move_threshold = move_threshold
        self.match_location_threshold = match_location_threshold
        self.low_feature_distance_threshold = low_feature_distance_threshold
        self.create_object_threshold = create_object_threshold
        
        self.metric = _nn_cosine_distance
        self.skip_norm = 50.0

    @staticmethod
    def get_visual_mat(objectfeatures, detectionfeatures, norm=True):
        visual_mat = np.zeros((len(objectfeatures), len(detectionfeatures)), dtype='float32')
        for i, obj in enumerate(objectfeatures):
            for j, det in enumerate(detectionfeatures):
                if norm:
                    # print("norm", obj.shape, det.shape)
                    obj_normalized = obj / np.linalg.norm(obj)
                    det_normalized = det / np.linalg.norm(det)
                    visual_mat[i, j] = np.linalg.norm(obj_normalized - det_normalized) / 2.0
        return visual_mat

    def get_iou_mat(self, objects, detections):
        iou_matrix = np.zeros((len(objects), len(detections)), dtype='float32')
        object_tlbrs = np.ascontiguousarray(np.array([o[1][-1].tlbr for o in objects], dtype=np.float64))
        detect_tlbrs = np.ascontiguousarray(np.array([[d['left'], d['top'], d['right'], d['bottom']] for d in detections], dtype=np.float64))
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
    
    @staticmethod
    def extract_matches_based_lapjv_with_vote(cost_matrix, threshold):
        if cost_matrix.size == 0:
            return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
        onum, dnum = cost_matrix.shape
        cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=threshold)
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
        unmatched_detection = [di for di in unmatched_detection if di != dnum - 1]
        matches = np.asarray(matches)
        return matches, unmatched_object, unmatched_detection

    @staticmethod
    def extract_low_features(image, hash_size=8):
        resized_img = cv2.resize(image, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
        gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
        mean = np.mean(gray_img)
        return np.where(gray_img > mean, 1, 0).flatten()

    @staticmethod
    def extract_low_feature_distance(lfeature1, lfeature2, hash_size):
        return np.sum(lfeature1 != lfeature2) / (hash_size * hash_size)

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
        cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=threshold)
        matches, unmatched_object, unmatched_detection = [], [], []
        for ix, mx in enumerate(x):
            if mx >= 0:
                matches.append([ix, mx])
        unmatched_object = np.where(x < 0)[0]
        unmatched_detection = np.where(y < 0)[0]
        matches = np.asarray(matches)
        return matches, unmatched_object, unmatched_detection

    def extract_features(self, ims):
        ims = [self.transform(Image.fromarray(im)) for im in ims]
        ims = torch.stack(ims).cuda()
        with torch.no_grad():
            high_dim_vectors = self.appearance_model(ims).cpu().numpy()
   
        return high_dim_vectors

    @staticmethod
    def compare_images(image_path1, image_path2, hash_size=8, hash_func=imagehash.average_hash):
        img1 = Image.open(image_path1)
        img2 = Image.open(image_path2)

        hash1 = hash_func(img1, hash_size=hash_size)
        hash2 = hash_func(img2, hash_size=hash_size)

        distance = hash1 - hash2

        return distance

    def get_gnncnn_mat(self, objects, detections, img_shape, frame_idx, normal_skip):
        width, height = img_shape
        input_nodes, input_edges, input_crops, input_edge_attrs = [], [], [], []
        senders, receivers = [], []
        for i, (objid, obj, _) in enumerate(objects):
            skip = frame_idx - obj[-1].last_real.frame_idx
            cx, cy = get_loc(obj[-1].d, width, height)
            input_nodes.append([cx, cy, (obj[-1].d['right']-obj[-1].d['left'])/width,
                                (obj[-1].d['bottom']-obj[-1].d['top'])/height, 1, 0, 0, skip / self.skip_norm])

            input_crops.append(self.transform(Image.fromarray(obj[-1].last_real.d['crop'])))
        input_nodes.append([0.5, 0.5, 0, 0, 0, 1, 0, normal_skip / self.skip_norm])
        input_crops.append(torch.zeros(3, SIZE_CROP, SIZE_CROP))
        for i, d in enumerate(detections):
            cx, cy = get_loc(d, width, height)
            input_nodes.append([cx, cy, (d['right']-d['left'])/width,
                                (d['bottom']-d['top'])/height, 0, 0, 1, normal_skip / self.skip_norm])
            input_crops.append(self.transform(Image.fromarray(d['crop'])))
        
        for i, (objid, obj, _) in enumerate(objects):
            detection1 = obj[-1].d
            x1, y1 = get_loc(detection1, width, height)
            for j, d in enumerate(detections):
                x2, y2 = get_loc(d, width, height)
                senders.extend([i, len(objects)+1+j])
                receivers.extend([len(objects)+1+j, i])
                input_edges.extend([[i, len(objects)+1+j], [len(objects)+1+j, i]])
                edge_shared = [x2 - x1, y2 - y1, math.sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1))]
                input_edge_attrs.extend([edge_shared + [1, 0, 0], edge_shared + [0, 1, 0]])
            senders.extend([i, len(objects)])
            receivers.extend([len(objects), i])
            input_edges.extend([[i, len(objects)], [len(objects), i]])
            input_edge_attrs.extend([[0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]])
        
        for i, (objid1, obj1, _) in enumerate(objects):
            detection1 = obj1[-1].d
            x1, y1 = get_loc(detection1, width, height)

            for j, (objid2, obj2, _) in enumerate(objects):
                if i == j:
                    continue

                detection2 = obj2[-1].d
                x2, y2 = get_loc(detection2, width, height)

                senders.append(i)
                receivers.append(j)
                input_edges.append([i, j])
                input_edge_attrs.append([x2 - x1, y2 - y1, math.sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)), 0, 0, 1])
        
        offset = len(objects) + 1
        for i, detection1 in enumerate(detections):
            x1, y1 = get_loc(detection1, width, height)
            for j, detection2 in enumerate(detections):
                if i == j:
                    continue
                x2, y2 = get_loc(detection2, width, height)

                senders.append(i+offset)
                receivers.append(j+offset)
                input_edges.append([i+offset, j+offset])
                input_edge_attrs.append([x2 - x1, y2 - y1, math.sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)), 0, 0, 1])
        
        input_crops = torch.stack(input_crops)
        input_nodes = torch.tensor(input_nodes, dtype=torch.float)
        input_edges = torch.tensor(input_edges, dtype=torch.long).t().contiguous()
        input_edge_attrs = torch.tensor(input_edge_attrs, dtype=torch.float)
        
        GraphData = Data(x=input_nodes, edge_index=input_edges, edge_attr=input_edge_attrs)
        GraphData = GraphData.to(self.device)
        input_crops = input_crops.to(self.device)
        
        with torch.no_grad():
            outputs = self.gnncnn_model(GraphData, input_crops)
            outputs = torch.sigmoid(outputs).squeeze().cpu().numpy()
        
        mat = np.zeros((len(objects), len(detections)+1), dtype='float32')
        for i, sender in enumerate(senders):
            receiver = receivers[i]
            if sender >= len(objects) or receiver < len(objects):
                continue
            s_idx = sender
            if receiver == len(objects):
                r_idx = len(detections)
            else:
                r_idx = receiver - len(objects) - 1
            mat[s_idx, r_idx] = outputs[i]
        print(mat)
        return mat 
            
    def update(self, frame_idx, im, detections, orishape):
        orig_width, orig_height = orishape
        track_ids = [None]*len(detections)
  
        # cleanup frames that are now in future if needed
        if len(self.frames) > 0 and frame_idx < self.frames[-1]:
            for id in list(self.objects.keys()):
                self.objects[id] = [d for d in self.objects[id] if d.frame_idx < frame_idx]
                if len(self.objects[id]) == 0:
                    del self.objects[id]
            self.frames = [idx for idx in self.frames if idx < frame_idx]
        self.frames.append(frame_idx)
        
        if len(self.frames) >= 2: skip = self.frames[-1] - self.frames[-2]
        else: skip = 0

        # get images if utilize 
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
            for imi, image in enumerate(images):
                detections[imi]['crop'] = image

        # match each object with current frame
        objects = [(id, dlist, frame_idx - dlist[-1].last_real.frame_idx) for id, dlist in self.objects.items() if (frame_idx - dlist[-1].last_real.frame_idx) < self.max_lost_time]
    
        detection_ids = list(range(len(detections)))
        if len(objects) > 0 and len(detections) > 0:
            iou_mat = self.get_iou_mat(objects, detections)
            stop_object_ids, move_object_ids, candidate_detection_ids = self.extract_matches_based_lapjv(iou_mat, self.move_threshold)
        
            for object_id, detection_id in stop_object_ids:
                # object_low_feature, detection_low_feature = self.extract_low_features(objects[object_id][1][-1].last_real.image, self.hash_size), self.extract_low_features(images[detection_id], self.hash_size)
                # low_feature_distance = self.extract_low_feature_distance(object_low_feature, detection_low_feature, self.hash_size)
                # if low_feature_distance > self.low_feature_distance_threshold:
                #     move_object_ids = np.array(list(move_object_ids) + [object_id])
                #     candidate_detection_ids = np.array(list(candidate_detection_ids) + [detection_id])
                #     continue
                d = Detection(detections[detection_id], frame_idx, image=images[detection_id])
                self.objects[objects[object_id][0]][-1].move_state = MoveState.STOP
                self.objects[objects[object_id][0]].append(d)
                # self.object_crops[objects[object_id][0]] = image[detection_id]
                track_ids[detection_id] = objects[object_id][0]
   
            objects = [objects[oid] for oid in move_object_ids]
            detections = [detections[did] for did in candidate_detection_ids]
            detection_ids = candidate_detection_ids

        if len(objects) > 0:
            # print('objects:', len(objects), "details:", objects)
            mat = self.get_gnncnn_mat(objects, detections, [orig_width, orig_height], frame_idx, skip)
            # update objects based either on mat or gt
            matched_object_ids, unmatched_object_ids, unmatched_detection_ids = self.extract_matches_based_lapjv_with_vote(mat, self.match_location_threshold)
        
            for object_id, detection_id in matched_object_ids:
                # print(object_id, detection_id, matched_object_ids)
                original_detection_id = detection_ids[detection_id]
                d = Detection(detections[detection_id], frame_idx, image=images[original_detection_id])
                self.objects[objects[object_id][0]].append(d)
                # self.object_crops[objects[object_id][0]] = image[original_detection_id]
                track_ids[original_detection_id] = objects[object_id][0]
        
            objects = [objects[oid] for oid in unmatched_object_ids]
            detections = [detections[did] for did in unmatched_detection_ids]
            detection_ids = [detection_ids[did] for did in unmatched_detection_ids]

        # update unmatched objects
        for objectIns in objects:
            d = Detection(fake_d, frame_idx, last_real=objectIns[1][-1].last_real)
            self.objects[objectIns[0]].append(d)
  
        # create new objects
        for detection_id, detection in zip(detection_ids, detections):
            # print(detection_id, detection)
            if detection['score'] < self.create_object_threshold: continue
            original_detection_id = detection_id
            object_id = self.next_id
            self.next_id += 1
            d = Detection(detection, frame_idx, image=images[original_detection_id])
            track_ids[original_detection_id] = object_id
            self.objects[object_id] = [d]
            # self.object_crops[object_id] = image[original_detection_id]

        frame_confidence = 1.0

        return track_ids, frame_confidence

if __name__ == '__main__':
    stdin = sys.stdin.detach()
    trackers = {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MOTGraphModel(8, 6)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    transform = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

    color_dict = {}
    
    while True:
        packet = read_json(stdin)
        if packet is None:
            break
        id = packet['id']
        msg = packet['type']
        frame_idx = packet['frame_idx']

        if msg == 'end':
            continue
        
        im = read_im(stdin)
        im = im[:, :, ::-1]
        
        detections = packet['detections']
        if detections is None:
            detections = []

        if id not in trackers:
            trackers[id] = Tracker(model, transform, device)

        track_ids, frame_confidence = trackers[id].update(frame_idx, im, detections, packet['resolution'])
        d = {
            'outputs': track_ids,
            'conf': frame_confidence,
            't': {},
        }
        sys.stdout.write('json'+json.dumps(d)+'\n')
        sys.stdout.flush()