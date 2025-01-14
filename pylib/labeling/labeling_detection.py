import os
import cv2
import sys
import json
import torch
import argparse
import numpy as np
from alive_progress import alive_bar

def iou(box1, box2):
    left1, top1, right1, bottom1 = box1
    left2, top2, right2, bottom2 = box2
    area1 = (right1 - left1) * (bottom1 - top1)
    area2 = (right2 - left2) * (bottom2 - top2)
    left = max(left1, left2)
    top = max(top1, top2)
    right = min(right1, right2)
    bottom = min(bottom1, bottom2)
    if left >= right or top >= bottom:
        return 0.0
    else:
        inter = (right - left) * (bottom - top)
        return inter / (area1 + area2 - inter)

parse = argparse.ArgumentParser()
parse.add_argument('--data_root', type=str, default='/home/xyc/otif/otif-dataset/dataset')
parse.add_argument('--data_name', type=str, default='taipei')
parse.add_argument('--data_flag', type=str, default='train/seg-train/images')
parse.add_argument('--width', type=int, default=640)
parse.add_argument('--height', type=int, default=640)
parse.add_argument('--threshold', type=float, default=0.4)
parse.add_argument('--device', type=str, default='0')

args = parse.parse_args()

param_width, param_height = args.width, args.height
threshold = args.threshold

sys.path.remove('/home/xyc/strongotif')
# print(sys.path, file=open('syspath.txt', 'a'), flush=True)
sys.path.append('/home/xyc/otif/otif-dataset/yolov5')
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

imgsz = (param_width, param_height)
dnn, half = False, False
augment, visualize, agnostic_nms = False, False, True
max_det = 100
conf_thres, iou_thres = threshold, 0.4

data_path1 = os.path.join(args.data_root, 'yolov5', 'data/coco.yaml') # VehicleFamily.yaml
weight_path1 = os.path.join('yolov5x.pt') # ./weights/Detectors/general/yolov5x.pt

data_path2 = os.path.join(args.data_root, 'yolov5', 'data/VehicleFamily.yaml')
weight_path2 = os.path.join('./weights/Detectors/general/yolov5x.pt')

device = select_device(args.device)
model1 = DetectMultiBackend(weight_path1,
                            device=device,
                            dnn=dnn, 
                            data=data_path1, 
                            fp16=half)

model2 = DetectMultiBackend(weight_path2,
                            device=device,
                            dnn=dnn, 
                            data=data_path2, 
                            fp16=half)

name_dict1 = model1.names
name_inverse_dict1 = {v: k for k, v in name_dict1.items()}

name_dict2 = model2.names
name_inverse_dict2 = {v: k for k, v in name_dict2.items()}

vid_stride=1
class_names = ["car", "bus", "truck"]
classes1=[name_inverse_dict1[class_name] for class_name in class_names]
stride, names, pt = model1.stride, model1.names, model1.pt
model1.warmup(imgsz=(1 if pt or model1.triton else 1, 3, *imgsz))  # warmup

classes2=[name_inverse_dict2[class_name] for class_name in class_names]
model2.warmup(imgsz=(1 if pt or model2.triton else 1, 3, *imgsz))  # warmup

imageList = [os.path.join(args.data_root, args.data_name, args.data_flag, path_name) for path_name in os.listdir(os.path.join(args.data_root, args.data_name, args.data_flag)) if path_name.endswith('.jpg') or path_name.endswith('.png')]
dataset = LoadImages(imageList, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)

with alive_bar(len(imageList)) as bar:
    for path, im, im0s, vid_cap, s in dataset:
        pathID = path.split('/')[-1].split('.')[0]
        im = torch.from_numpy(im).to(model1.device)
        im = im.half() if model1.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        # Inference
        with torch.no_grad():
            pred1 = model1(im, augment=augment, visualize=visualize)
            pred2 = model2(im, augment=augment, visualize=visualize)

        # NMS
        pred1 = non_max_suppression(pred1, conf_thres, iou_thres, classes=classes1, agnostic=agnostic_nms, max_det=max_det)
        pred2 = non_max_suppression(pred2, conf_thres, iou_thres, classes=classes2, agnostic=agnostic_nms, max_det=max_det)
        
        detections = []
        for det1, det2 in zip(pred1, pred2):  # per image
            im0 = im0s.copy()

            if len(det1) or len(det2):
                # Rescale boxes from img_size to im0 size
                det1[:, :4] = scale_boxes(im.shape[2:], det1[:, :4], im0.shape).round()
                det2[:, :4] = scale_boxes(im.shape[2:], det2[:, :4], im0.shape).round()
                # Fuse detections
                fuse_matrix = np.zeros((len(det1), len(det2)))
                for i, (*x1y1x2y2, conf, cls) in enumerate(det1):
                    for j, (*x1y1x2y2, conf, cls) in enumerate(det2):
                        fuse_matrix[i, j] = iou(x1y1x2y2, x1y1x2y2)
                fuse_matrix = fuse_matrix > 0.9
                for i, (*x1y1x2y2, conf, cls) in enumerate(det1):
                    for j, (*x1y1x2y2, conf, cls) in enumerate(det2):
                        if fuse_matrix[i, j]:
                            det1[i, 4] = 0
                            break
                for i, (*x1y1x2y2, conf, cls) in enumerate(det1):
                    det1[i, 5] = int(name_inverse_dict2[name_dict1[int(cls)]])
                det1, det2 = det1.cpu().numpy(), det2.cpu().numpy()
                det = np.concatenate((det1, det2), axis=0)
                # Write results
                for *x1y1x2y2, conf, cls in reversed(det):
                    w, h = x1y1x2y2[2] - x1y1x2y2[0], x1y1x2y2[3] - x1y1x2y2[1]
                    if w * h < 500 or w * h > 200000 or conf < conf_thres:
                        continue
                    detections.append({"class": name_dict2[int(cls)], "score": float(conf), "left": int(x1y1x2y2[0]), "top": int(x1y1x2y2[1]), "right": int(x1y1x2y2[2]), "bottom": int(x1y1x2y2[3])})
                    cv2.rectangle(im0, (int(x1y1x2y2[0]), int(x1y1x2y2[1])), (int(x1y1x2y2[2]), int(x1y1x2y2[3])), (0, 255, 0), 2)
                    cv2.putText(im0, f'{name_dict2[int(cls)]} {conf:.2f}', (int(x1y1x2y2[0]), int(x1y1x2y2[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imwrite(f'./labelingTest/{pathID}.jpg', im0)
        print(os.path.join(args.data_root, args.data_name, args.data_flag, f'{pathID}.json'))
        json.dump(detections, open(os.path.join(args.data_root, args.data_name, args.data_flag, f'{pathID}.json'), 'w'))
        bar()