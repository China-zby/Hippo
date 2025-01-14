import os
import sys
import json
import torch
import numpy as np

sys.path.pop(-1)
# print(sys.path, file=open('syspath.txt', 'w'), flush=True)
sys.path.append('/mnt/data_ssd1/lzp/otif-dataset/yolov5')
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode
from utils.augmentations import (Albumentations, augment_hsv, classify_albumentations, classify_transforms, copy_paste,
                                 letterbox, mixup, random_perspective)

class Detector:
    def __init__(self, label, 
                 batch_size=1,
                 threshold=0.25, imgsz=(640, 640)):
        self.imgsz = imgsz
        dnn, half = False, False
        self.augment, self.visualize, self.agnostic_nms = False, False, False
        self.max_det = 20
        self.conf_thres, self.iou_thres = threshold, 0.4
        self.auto = True

        data_path = os.path.join('/home/xyc/otif/otif-dataset/yolov5', 'data/VehicleFamily.yaml')
        weight_path = os.path.join(f'/home/lzp/go-work/src/otifpipeline/weights/Detectors/{label}/yolov5n.pt')

        device = select_device('0')
        self.model = DetectMultiBackend(weight_path,
                                device=device,
                                dnn=dnn, 
                                data=data_path, 
                                fp16=half)
        name_dict = self.model.names
        name_inverse_dict = {v: k for k, v in name_dict.items()}

        self.classes = "car,bus,truck"
        self.classes = [name_inverse_dict[c] for c in self.classes.split(',')]
        self.stride, names, pt = self.model.stride, self.model.names, self.model.pt
        self.model.warmup(imgsz=(1 if pt or self.model.triton else batch_size, 3, *imgsz))  # warmup
        
    def __call__(self, im0, save_json_path=None, save_image_path=None, human=False):
        im = letterbox(im0, self.imgsz, stride=self.stride, auto=self.auto)[0]  # padded resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous
        
        im = torch.from_numpy(im).to(self.model.device)
        im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        
        pred = self.model(im, augment=self.augment, visualize=self.visualize)
        pred = non_max_suppression(pred, 
                                   self.conf_thres, 
                                   self.iou_thres, 
                                   self.classes, 
                                   self.agnostic_nms,
                                   max_det=self.max_det)
        
        # Process predictions
        det = pred[0]
        global detections, new_detections
        detections = []
        if len(det):
            # Write results
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                detections.append({
                        'class': self.model.names[int(cls)],
                        'score': float(conf),
                        'left': int(xyxy[0]),
                        'right': int(xyxy[2]),
                        'top': int(xyxy[1]),
                        'bottom': int(xyxy[3]),
                    })
            new_detections = detections.copy()
            
            def mouse_callback(event, x, y, flags, param):
                global detections, new_detections

                # 如果单击鼠标，则检查鼠标是否在一个或多个矩形内，如果是，则删除该矩形
                if event == cv2.EVENT_LBUTTONDOWN:
                    new_detections = []
                    delete = False
                    for rect in detections:
                        if x > rect["left"] and x < rect["right"] and y > rect["top"] and y < rect["bottom"] and not delete:
                            delete = True
                            continue
                        new_detections.append(rect)
                    detections = new_detections
            
            cv2.namedWindow("image")
            cv2.setMouseCallback("image", mouse_callback)
            
            while True:
                im0copy = im0.copy()
                for rect in detections:
                    cv2.rectangle(im0copy, (rect["left"], rect["top"]), (rect["right"], rect["bottom"]), (0, 255, 0), 2)
                cv2.namedWindow("image")
                cv2.setMouseCallback("image", mouse_callback)
                cv2.imshow("image", im0copy)
                key = cv2.waitKey(1)

                # 按下ESC键退出循环
                if key == 27:
                    break

                # 如果某些矩形已被删除，则刷新窗口以删除这些矩形
                if len(detections) < len(new_detections):
                    im0copy = im0.copy()
                    for rect in detections:
                        cv2.rectangle(im0copy, (rect["left"], rect["top"]), (rect["right"], rect["bottom"]), (0, 255, 0), 2)
            
        # print('json'+json.dumps(detections), flush=True)
        if save_json_path is not None:
            with open(save_json_path, 'w') as f:
                json.dump(detections, f)
                
        if save_image_path is not None:
            cv2.imwrite(save_image_path, im0)
                
        