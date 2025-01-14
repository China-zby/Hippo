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

# os.environ["CUDA_VISIBLE_DEVICES"]=f"{device_id}"

# os.chdir(os.path.join(data_root, 'yolov5/'))
# sys.path.append('./')
# print(sys.path, file=open('syspath.txt', 'a'), flush=True)
# sys.path.remove('/home/xyc/strongotif')
# print(sys.path, file=open('syspath.txt', 'a'), flush=True)
sys.path.append(os.path.join(data_root, 'yolov5/'))
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

seen = 0
imgsz = (param_width, param_height)
dnn, half = False, False
augment, visualize, agnostic_nms = False, False, False
conf_thres, iou_thres = threshold, 0.4

data_path = os.path.join(data_root, 'yolov5', 'data/coco.yaml')
weight_path = os.path.join(f'./weights/Detectors/general/yolov5{modelsize.lower()}.pt')

device = select_device(f'{device_id}')
model = DetectMultiBackend(weight_path,
                           device=device,
                           dnn=dnn, 
                           data=data_path, 
                           fp16=half)
name_dict = model.names
name_inverse_dict = {v: k for k, v in name_dict.items()}

classes = [name_inverse_dict[c] for c in classes.split(',')]
stride, names, pt = model.stride, model.names, model.pt
model.warmup(imgsz=(1 if pt or model.triton else batch_size, 3, *imgsz))  # warmup

stdin = sys.stdin.detach()
timei = 0
while True:
    timei += 1
    im = read_im(stdin)
    if im is None:
        break
    # im0 = im[0].copy()
    # print('im', im.shape, f'/home/lzp/go-work/src/otifpipeline/test/{timei}.jpg', file=open('imshape.txt', 'a'), flush=True)
    im = im.transpose((0, 3, 1, 2))
    im = torch.from_numpy(im).to(model.device)
    im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
    im /= 255
    if len(im.shape) == 3:
        im = im[None]
        
    pred = model(im, augment=augment, visualize=visualize)
    pred = non_max_suppression(pred, 
                               conf_thres, 
                               iou_thres, 
                               classes, 
                               agnostic_nms)

    # Process predictions
    detections = []
    for i, det in enumerate(pred):  # per image
        seen += 1

        dlist = []
        if len(det):
            # Write results
            for *xyxy, conf, cls in reversed(det):
                dlist.append({
                        'class': model.names[int(cls)],
                        'score': float(conf),
                        'left': int(xyxy[0]),
                        'right': int(xyxy[2]),
                        'top': int(xyxy[1]),
                        'bottom': int(xyxy[3]),
                    })
                # cv2.rectangle(im0, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
        detections.append(dlist)
    detections = [nms(detection, 0.1) for detection in detections]
    # cv2.imwrite(f'/home/lzp/go-work/src/otifpipeline/test/{timei}.jpg', im0)
    print('json'+json.dumps(detections), flush=True)