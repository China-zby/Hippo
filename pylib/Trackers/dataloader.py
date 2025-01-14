import os
import math
import json
import random
import struct

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torch_geometric.data import Data, Batch

SCENEDICT = {
    "amsterdam": 0,
    "warsaw": 1, "shibuya": 2,
    "jackson": 3, "caldot1": 4,
    "caldot2": 5, 'uav': 6
}


def get_loc(detection, ORIG_WIDTH, ORIG_HEIGHT):
    cx = (detection['left'] + detection['right']) / 2
    cy = (detection['top'] + detection['bottom']) / 2
    cx = float(cx) / ORIG_WIDTH
    cy = float(cy) / ORIG_HEIGHT
    return cx, cy


def get_fake_d(NORM):
    fake_d = {
        'left': -NORM,
        'top': -NORM,
        'right': -NORM,
        'bottom': -NORM,
    }
    return fake_d


def read_im(stdin):
    buf = stdin.read(12)
    if not buf:
        return None
    (l, width, height) = struct.unpack('>III', buf)
    buf = stdin.read(l)
    return np.frombuffer(buf, dtype='uint8').reshape((height, width, 3))


def read_json(stdin):
    buf = stdin.read(4)
    if not buf:
        return None
    (l,) = struct.unpack('>I', buf)
    buf = stdin.read(l)
    return json.loads(buf.decode('utf-8'))


def clip(x, lo, hi):
    if x < lo:
        return lo
    elif x > hi:
        return hi
    else:
        return x


def repr_detection(t, d, NORM):
    return [
        d['left']/NORM,
        d['top']/NORM,
        d['right']/NORM,
        d['bottom']/NORM,
        float(t)/32.0,
    ]


def skip_onetime(skip_list):
    skip = random.choice(skip_list)
    return skip


def get_frame_info(detections, label, frame_idx, frame_dir, original_size):
    width, height = original_size
    frame_path = os.path.join(frame_dir, f'{frame_idx:06d}.jpg')
    frame_info = [frame_path]
    for detection_idx, detection in enumerate(detections):
        if detection['class'] != label:
            continue
        detection['width'], detection['height'] = float(
            detection['right'] - detection['left']) / width, float(detection['bottom'] - detection['top']) / height

        frame_info.append((detection, detection_idx))
    return frame_info


def read_image(image_path):
    image = Image.open(image_path)
    return image


def get_crop(image, detection):
    crop = image.crop(
        (detection['left'], detection['top'], detection['right'], detection['bottom']))
    return crop


def get_frame_pair(info1, info2, skip, oriSize, cropSize, skip_norm=50.0):
    # we need to get two graphs:
    # * input: (pos, features) for each node, (delta, distance, reverse) for each edge
    # * target: (match) for each edge
    # node <len(info1)> is special, it's for detections in cur frame that don't match any in next frame

    width, height = oriSize
    input_nodes = []
    input_edges = []
    input_crops = []
    input_edge_attrs = []
    target_edge_attrs = []

    for i, t in enumerate(info1):
        detection, crop, _ = t
        cx, cy = get_loc(detection, width, height)
        input_nodes.append([cx, cy, detection['width'],
                           detection['height'], 1, 0, 0, skip/skip_norm])
        input_crops.append(crop)
    input_nodes.append([0.5, 0.5, 0, 0, 0, 1, 0, skip/skip_norm])
    input_crops.append(torch.zeros((3, cropSize, cropSize)))
    for i, t in enumerate(info2):
        detection, crop, _ = t
        cx, cy = get_loc(detection, width, height)
        input_nodes.append([cx, cy, detection['width'],
                           detection['height'], 0, 0, 1, skip/skip_norm])
        input_crops.append(crop)

    num_matches = 0
    for i, t1 in enumerate(info1):
        detection1, _, _ = t1
        x1, y1 = get_loc(detection1, width, height)
        does_match = False

        for j, t2 in enumerate(info2):
            detection2, _, _ = t2
            x2, y2 = get_loc(detection2, width, height)

            input_edges.append([i, len(info1) + 1 + j])
            input_edges.append([len(info1) + 1 + j, i])

            edge_shared = [x2 - x1, y2 - y1,
                           math.sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1))]

            input_edge_attrs.append(edge_shared + [1, 0, 0])
            input_edge_attrs.append(edge_shared + [0, 1, 0])

            if detection1['track_id'] == detection2['track_id']:
                label = 1.0
                does_match = True
            else:
                label = 0.0
            target_edge_attrs.append([label])
            target_edge_attrs.append([0.0])

        input_edges.append([i, len(info1)])
        input_edges.append([len(info1), i])

        edge_shared = [0.0, 0.0, 0.0]
        input_edge_attrs.append(edge_shared + [0, 0, 0])
        input_edge_attrs.append(edge_shared + [1, 0, 0])

        if does_match:
            label = 0.0
            num_matches += 1
        else:
            label = 1.0
        target_edge_attrs.append([label])
        target_edge_attrs.append([0.0])

    if num_matches == 0 and False:
        return None, None, None

    def add_internal_edges(info, offset):
        for i, t1 in enumerate(info):
            detection1, _, _ = t1
            x1, y1 = get_loc(detection1, width, height)

            for j, t2 in enumerate(info):
                if i == j:
                    continue

                detection2, _, _ = t2
                x2, y2 = get_loc(detection2, width, height)

                input_edges.append([offset + i, offset + j])
                input_edge_attrs.append(
                    [x2 - x1, y2 - y1, math.sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1))] + [0, 0, 1])
                target_edge_attrs.append([0.0])

    add_internal_edges(info1, 0)
    add_internal_edges(info2, len(info1) + 1)

    input_nodes = torch.tensor(input_nodes, dtype=torch.float)
    input_edges = torch.tensor(input_edges, dtype=torch.long).t().contiguous()
    input_edge_attrs = torch.tensor(input_edge_attrs, dtype=torch.float)
    target_edge_attrs = torch.tensor(target_edge_attrs, dtype=torch.float)

    input_dict = {
        "x": input_nodes,
        "edge_index": input_edges,
        "edge_attr": input_edge_attrs,
        "y": target_edge_attrs
    }
    return input_dict, input_crops, num_matches


def datasample(videos,
               skips,
               fps,
               info_dir=None,
               NORM=None,
               MAX_LENGTH=None,
               NUM_BOXES=None):
    fake_d = {
        'left': -NORM,
        'top': -NORM,
        'right': -NORM,
        'bottom': -NORM,
    }

    my_skips = skips

    track_length_category = random.randint(0, 2)
    # skip_rng = random.randint(1, len(my_skips))
    skip_rng = 1
    skip_idx = random.randint(0, len(my_skips)-skip_rng)
    sample_skips = my_skips[skip_idx:skip_idx+skip_rng]

    while True:
        detections, tracks, videoid = random.choice(videos)

        # videoinfopath = os.path.join(info_dir, f'{videoid}.txt')
        # with open(videoinfopath, 'r') as f:
        #     videoinfo = f.readlines()
        # videoinfo = videoinfo[0].split('-')
        sceneid = [0] # SCENEDICT[videoinfo[0]]

        input_sceneids = np.asarray(sceneid)

        if track_length_category == 1:
            tracks = [t for t in tracks if len(t[1]) >= fps]
        elif track_length_category == 2:
            tracks = [t for t in tracks if len(t[1]) >= 2*fps]

        if not tracks:
            continue

        track_id, dlist = random.choice(tracks)
        start_frame, start_idx = random.choice(dlist[0:1+len(dlist)//2])

        inputs = np.zeros((MAX_LENGTH, 10), dtype='float32')
        inputs[0, 0:5] = repr_detection(
            0, detections[start_frame][start_idx], NORM)
        inputs[0, 5:10] = repr_detection(
            0, detections[start_frame][start_idx], NORM)
        boxes = np.zeros((MAX_LENGTH, NUM_BOXES, 5), dtype='float32')
        boxes[:, :, :] = -2
        boxes[:, 0, :] = -1
        targets = np.zeros((MAX_LENGTH, NUM_BOXES), dtype='float32')
        targets[:, 0] = 1
        mask = np.zeros((MAX_LENGTH,), dtype='float32')

        last_d = detections[start_frame][start_idx]
        last_frame = start_frame
        cur_skip = random.choice(sample_skips)  # skip from last input
        frame_idx = start_frame + cur_skip
        i = 0
        while frame_idx < len(detections) and i < MAX_LENGTH-1 and frame_idx-last_frame < 20+cur_skip:
            # pre-fill input with fake detection in case we don't find the right one
            inputs[i+1, 0:5] = repr_detection(cur_skip, fake_d, NORM)

            dlist = []
            if detections[frame_idx]:
                mask[i] = 1
                dlist = detections[frame_idx]

                if len(dlist) > NUM_BOXES-1:
                    good = [d for d in dlist if d['track_id'] == track_id]
                    bad = [d for d in dlist if d['track_id'] != track_id]
                    dlist = good + random.sample(bad, NUM_BOXES-1-len(good))
            for det_idx, d in enumerate(dlist):
                boxes[i, det_idx+1, :] = repr_detection(cur_skip, d, NORM)
                if d['track_id'] == track_id:
                    inputs[i+1, 0:5] = repr_detection(cur_skip, d, NORM)
                    last_d = d
                    last_frame = frame_idx
                    targets[i, 0] = 0
                    targets[i, det_idx+1] = 1

            inputs[i+1,
                   5:10] = repr_detection(frame_idx - last_frame, last_d, NORM)
            cur_skip = random.choice(sample_skips)
            frame_idx += cur_skip
            i += 1

        if i == 0 or mask.max() == 0:
            continue

        return inputs, boxes, mask, targets, input_sceneids


class MOTGraphDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, skip_list, width, height, crop_size):
        super(MOTGraphDataset, self).__init__()
        self.dataset_path = dataset_path
        self.skip_list = skip_list
        self.load_skip = 1
        self.labels = ['car', 'bus', 'truck']
        self.width, self.height = width, height
        self.crop_size = crop_size
        self.data_list = self.load_data()
        self.transform = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def load_data(self):
        # Load the dataset from the dataset_path and
        # convert it into a list of torch_geometric.data.Data objects
        video_dir = os.path.join(self.dataset_path, 'frames')
        track_dir = os.path.join(self.dataset_path, 'tracks')
        video_path_name_list = os.listdir(video_dir)
        dataset_list = []
        for label in self.labels:
            for video_id in tqdm(video_path_name_list):
                frame_dir = os.path.join(video_dir, video_id)
                detection_path = os.path.join(track_dir, f'{video_id}.json')

                with open(detection_path, 'r') as f:
                    detections = json.load(f)

                for frame_idx in range(0, len(detections), self.load_skip):
                    sample_skip_number = skip_onetime(self.skip_list)
                    if frame_idx + sample_skip_number >= len(detections) or not detections[frame_idx] or not detections[frame_idx + sample_skip_number]:
                        continue

                    frame_info_left = get_frame_info(detections[frame_idx],
                                                     label,
                                                     frame_idx,
                                                     frame_dir, [self.width, self.height])
                    frame_info_right = get_frame_info(detections[frame_idx + sample_skip_number],
                                                      label,
                                                      frame_idx + sample_skip_number,
                                                      frame_dir, [self.width, self.height])

                    if len(frame_info_left) == 1 or len(frame_info_right) == 1 or len(frame_info_left) + len(frame_info_right) < 4:
                        continue

                    dataset_list.append(
                        (frame_info_left, frame_info_right, sample_skip_number))

        return dataset_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        frame_info_left, frame_info_right, sample_skip_number = self.data_list[idx]
        frame_image_left, frame_image_right = read_image(
            frame_info_left[0]), read_image(frame_info_right[0])
        frame_info_with_crop_left, frame_info_with_crop_right = [], []
        for detection, detection_idx in frame_info_left[1:]:
            frame_info_with_crop_left.append((detection, self.transform(
                get_crop(frame_image_left, detection)), detection_idx))
        for detection, detection_idx in frame_info_right[1:]:
            frame_info_with_crop_right.append((detection, self.transform(
                get_crop(frame_image_right, detection)), detection_idx))
        input_dict, input_crops, num_matches = get_frame_pair(frame_info_with_crop_left,
                                                              frame_info_with_crop_right,
                                                              sample_skip_number,
                                                              [self.width,
                                                                  self.height],
                                                              self.crop_size)

        input_crops = torch.stack(input_crops)

        return input_dict, input_crops, num_matches

    def collate_fn(self, batch):
        input_dict_list, target_dict_list, input_crops_list = [], [], []
        for input_dict, input_crops, _ in batch:
            # print(input_dict['nodes'], input_dict['nodes'].shape)
            # print(input_dict['edges'], input_dict['edges'].shape)
            # print(input_dict['x'].shape, input_dict['edge_index'].shape, input_dict['edge_attr'].shape, input_dict['y'].shape)
            input_graph = Data(x=input_dict['x'], edge_index=input_dict['edge_index'],
                               edge_attr=input_dict['edge_attr'], y=input_dict['y'])
            # print(input_graph, input_crops.shape)
            input_dict_list.append(input_graph)
            # target_dict_list.append(target_dict)
            input_crops_list.append(input_crops)
            # num_matches_list.append(num_matches)
        input_dict_list = Batch.from_data_list(input_dict_list)
        # target_dict_list = Batch.from_data_list(target_dict_list)
        input_crops_list = torch.cat(input_crops_list)
        # num_matches_list = torch.tensor(num_matches_list)
        return input_dict_list, input_crops_list  # , num_matches_list
