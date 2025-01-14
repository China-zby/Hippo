
import os
import sys

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
if os.path.join(data_root, 'yolov7/') not in sys.path:
    sys.path.append(os.path.join(data_root, 'yolov7/'))
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, \
    scale_coords

imgsz = (param_width, param_height)
dnn, half = False, False
augment, visualize, agnostic_nms = False, False, False
max_det = 20
conf_thres, iou_thres = threshold, 0.4

device = torch.device(f"cuda:{device_id}")
model = attempt_load(
    f'./weights/Detectors/general/yolov7{modelsize.lower()}.pt', map_location=device)  # load FP32 model
stride = int(model.stride.max())

if half:
    model.half()

name_dict = model.module.names if hasattr(model, 'module') else model.names

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
    refine_imgsz = [check_img_size(im.shape[1], s=stride),
                    check_img_size(im.shape[0], s=stride)]
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

    with torch.no_grad():
        pred = model(im, augment=augment)[0]
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
                det[:, :4] = scale_coords(im.shape[2:],  # new image
                                          det[:, :4],
                                          # orginal image
                                          list(reversed(imgsz))
                                          ).round()  # w, h
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
