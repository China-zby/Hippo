from ultralytics import YOLO
import os
import cv2
import sys
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


imgsz = (param_height, param_width)
dnn, half = False, False
conf_thre, iou_thre = threshold, 0.4

device = torch.device(f"cuda:{device_id}")
model = YOLO(f'./weights/Detectors/general/yolov8{modelsize.lower()}.pt')
model.model = model.model.cuda()
model.model.eval()
if half:
    model.half()

name_dict = model.names
name_inverse_dict = {v: k for k, v in name_dict.items()}

classes = [name_inverse_dict[c] for c in classes.split(',')]
# _ = model.predict(torch.zeros((1, 3, *imgsz), device=device), verbose=False)  # warmup

stdin = sys.stdin.detach()
timei = 0
while True:
    im = read_im(stdin)  # batchsize, height, width, 3
    if im is None:
        break
    im = im.transpose((0, 3, 1, 2))  # batchsize, 3, height, width
    im = torch.from_numpy(im).to(model.device)  # batchsize, 3, height, width
    im = im.half() if half else im.float()  # uint8 to fp16/32
    im /= 255  # 0-255 to 0-1
    if len(im.shape) == 3:
        im = im[None]

    pred = model.predict(im, imgsz=imgsz, conf=conf_thre,
                         iou=iou_thre, classes=classes,
                         agnostic_nms=False,
                         verbose=False)

    # Process predictions
    detections = []
    for i, det in enumerate(pred):  # per image
        dlist = []
        if len(det):
            for xyxy, conf, cls in zip(det.boxes.xyxy, det.boxes.conf, det.boxes.cls):
                dlist.append({
                    'class': model.names[int(cls)],
                    'score': float(conf),
                    'left': int(xyxy[0]),
                    'right': int(xyxy[2]),
                    'top': int(xyxy[1]),
                    'bottom': int(xyxy[3]),
                })
        detections.append(dlist)
    detections = [nms(detection, 0.1) for detection in detections]
    print('json'+json.dumps(detections), flush=True)
