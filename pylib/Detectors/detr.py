import os
import cv2
import sys
import json
import torch
import numpy as np
from PIL import Image
from dataloader import read_im
import torchvision.transforms as T
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

os.environ["CUDA_VISIBLE_DEVICES"]=f"{device_id}"

model = torch.hub.load('facebookresearch/detr:main', f'detr_resnet{modelsize}', pretrained=True)
model.eval()
model.cuda()

if not "names" in dir(model):
    model.names = {3:"car", 6:"bus", 8:"truck"}

transform = T.Compose([
    T.ToTensor(),
    T.Resize((800, 800)),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# im0 = cv2.imread('out_test.jpeg')

stdin = sys.stdin.detach()
timei = 0
while True:
    timei += 1
    rawims = read_im(stdin)
    # rawims = [im0]
    if rawims is None:
        break
    im0 = rawims[0]
    ims = []
    for im in rawims:
        ims.append(transform(Image.fromarray(im)).unsqueeze(0))
    ims = torch.cat(ims, dim=0).cuda()
        
    with torch.no_grad():
        outputs = model(ims)

    # print(outputs['pred_logits'].softmax(-1).shape, outputs['pred_boxes'].shape)
    
    detections = []
    for probas, boxes in zip(outputs['pred_logits'], outputs['pred_boxes']):
        probas = probas.softmax(-1)
        probas = probas[:, :-1]
        
        keep = probas.max(-1).values > threshold

        probas_to_keep = probas[keep]
        boxes_to_keep = boxes[keep]
        
        # print(probas.shape, boxes.shape, probas_to_keep.shape, boxes_to_keep.shape)
        
        # print(im0.shape, file=open('out.txt', 'a'))
        img_width, img_height = im0.shape[1], im0.shape[0]

        pixel_boxes = boxes_to_keep.cpu().numpy() * np.array([img_width, img_height, img_width, img_height])

        dlist = []
        for box, score in zip(pixel_boxes, probas_to_keep):
            # print(score.max(), score.argmax(), box)
            if score.max() > threshold and int(score.argmax()) in model.names.keys():
                x_center, y_center, w, h = box
                dlist.append({
                    'class': model.names[int(score.argmax())],
                    'score': float(score.max()),
                    'left': int(x_center - w / 2),
                    'right': int(x_center + w / 2),
                    'top': int(y_center - h / 2),
                    'bottom': int(y_center + h / 2),
                })
        detections.append(dlist)
        
    detections = [nms(detection, 0.1) for detection in detections]
    
    print('json'+json.dumps(detections), flush=True)