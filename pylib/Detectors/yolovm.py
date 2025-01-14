
import os
import sys
import math
from typing import List, Optional

import cv2
import json
import torch
import numpy as np
from dataloader import read_im
from cython_bbox import bbox_overlaps as bbox_ious

# Function to convert detection dictionary to box format [x1, y1, x2, y2]


def get_boxes(detections):
    boxes = []
    for det in detections:
        boxes.append([det['left'], det['top'], det['right'], det['bottom']])
    return boxes

# Non-Maximum Suppression Function


def nms(detections, iou_threshold):
    if len(detections) == 0:
        return []

    detections = sorted(detections, key=lambda x: x['score'], reverse=True)

    boxes = get_boxes(detections)
    ious = bbox_ious(np.array(boxes, dtype=np.float),
                     np.array(boxes, dtype=np.float))

    keep = [True]*len(detections)

    for i in range(len(detections)):
        if not keep[i]:
            continue
        for j in range(i+1, len(detections)):
            if keep[j] and ious[i, j] > iou_threshold:
                keep[j] = False

    return [det for k, det in zip(keep, detections) if k]


data_root = sys.argv[1]
batch_size = int(sys.argv[2])
param_width = int(sys.argv[3])
param_height = int(sys.argv[4])
threshold = float(sys.argv[5])
classes = sys.argv[6]
label = sys.argv[7]
modelsize = sys.argv[8]
device_id = sys.argv[9]

# Change directory so that imports wortk correctly
if os.path.join(data_root, 'YOLOv6/') not in sys.path:
    sys.path.append(os.path.join(data_root, 'YOLOv6/'))
from yolov6.core.inferer import Inferer
from yolov6.utils.nms import non_max_suppression
from yolov6.data.data_augment import letterbox
from yolov6.layers.common import DetectBackend
from yolov6.utils.events import LOGGER, load_yaml

def check_img_size(img_size, s=32, floor=0):
    def make_divisible(x, divisor):
        # Upward revision the value x to make it evenly divisible by the divisor.
        return math.ceil(x / divisor) * divisor
    """Make sure image size is a multiple of stride s in each dimension, and return a new shape list of image."""
    if isinstance(img_size, int):  # integer i.e. img_size=640
        new_size = max(make_divisible(img_size, int(s)), floor)
    elif isinstance(img_size, list):  # list i.e. img_size=[640, 480]
        new_size = [max(make_divisible(x, int(s)), floor) for x in img_size]
    else:
        raise Exception(f"Unsupported type of img_size: {type(img_size)}")

    if new_size != img_size:
        print(
            f'WARNING: --img-size {img_size} must be multiple of max stride {s}, updating to {new_size}')
    return new_size if isinstance(img_size, list) else [new_size]*2


imgsz = [param_width, param_height]
dnn, half = False, False
augment, visualize, agnostic_nms = False, False, False
max_det = 20
conf_thres, iou_thres = threshold, 0.4

device = torch.device(f"cuda:{device_id}")
modelsizelower = modelsize.lower()
# if modelsizelower in ['ll', 'lm', 'ls']:
#     modelweightname = f'lite{modelsizelower[-1]}'
# else:
#     modelweightname = modelsizelower
model = DetectBackend(
    f'./weights/Detectors/general/yolov6{modelsizelower}.pt', device=device)

stride = model.stride
name_dict = load_yaml("./weights/coco.yaml")['names']

name_inverse_dict = {v: k for k, v in enumerate(name_dict)}
classes = [name_inverse_dict[c] for c in classes.split(',')]

if half & (device.type != 'cpu'):
    model.model.half()
else:
    model.model.float()
    half = False
model.model.eval()

# if device.type != 'cpu':
#     model(torch.zeros(1, 3, max(imgsz), max(imgsz)).to(
#         device).type_as(next(model.model.parameters())))  # warmup

stdin = sys.stdin.detach()
while True:
    im = read_im(stdin)
    if im is None:
        break

    resize_flag = False
    refine_imgsz = check_img_size(
        [int(im.shape[1]), int(im.shape[0])], s=stride)
    if refine_imgsz[0] != int(im.shape[1]) or refine_imgsz[1] != int(im.shape[0]):
        im = cv2.cvtColor(im[0], cv2.COLOR_RGB2BGR)
        im = letterbox(im, max(refine_imgsz), stride=stride)[0]
        im = im[:, :, ::-1]
        im = np.expand_dims(im, axis=0)
        resize_flag = True

    im = im.transpose((0, 3, 1, 2))
    im = np.ascontiguousarray(im)
    im = torch.from_numpy(im).to(device)  # batchsize, 3, height, width
    im = im.half() if half else im.float()  # uint8 to fp16/32
    im /= 255
    if len(im.shape) == 3:
        im = im[None]

    with torch.no_grad():
        pred = model(im)
        pred = non_max_suppression(pred,
                                   conf_thres,
                                   iou_thres,
                                   classes,
                                   agnostic_nms,
                                   max_det=max_det)

    # Process predictions
    detections = []
    for i, det in enumerate(pred):  # per image
        dlist = []
        if len(det):
            if resize_flag:
                det[:, :4] = Inferer.rescale(
                    im.shape[2:], det[:, :4], list(reversed(imgsz))).round()
            # Write results
            for *xyxy, conf, cls in reversed(det):
                dlist.append({
                    'class': name_dict[int(cls)],
                    'score': float(conf),
                    'left': int(xyxy[0]),
                    'right': int(xyxy[2]),
                    'top': int(xyxy[1]),
                    'bottom': int(xyxy[3]),
                })
        detections.append(dlist)
    detections = [nms(detection, 0.1) for detection in detections]
    print('json'+json.dumps(detections), flush=True)
