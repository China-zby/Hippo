import os
import cv2
import sys
import json
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

os.environ["CUDA_VISIBLE_DEVICES"]=f"{device_id}"

classes = classes.split(",")

config = "/home/xyc/otif/otif-dataset/Image-Adaptive-YOLO/data/classes/vocdark.names"
checkpoint = "./weights/Detectors/general/lowlight/yolov3_test_loss=9.7815.ckpt-62"

sys.path.pop(-1)
# sys.path.remove('/home/xyc/strongotif')
sys.path.append(os.path.join(data_root, 'Image-Adaptive-YOLO/'))

# print("sys path", sys.path, file=open('/home/lzp/go-work/src/otifpipeline/input_size.txt', 'a'))

from evaluate_lowlight import YoloTest

# input_size = (param_height, param_width)
detector = YoloTest(checkpoint, config, threshold)
# print(detector, file=open('/home/lzp/go-work/src/otifpipeline/input_size.txt', 'a'))
stdin = sys.stdin.detach()
while True:
    imgs = read_im(stdin)
    if imgs is None:
        break
    detections = []
    for im in imgs: 
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        max_width_height = max(im.shape[0], im.shape[1])
        input_size = [max_width_height, max_width_height] # [544, 544]
        # print(input_size, file=open('/home/lzp/go-work/src/otifpipeline/input_size.txt', 'a'))
        bboxes = detector.predict(im, input_size)[0]
        dlist = []
        for bbox in bboxes:
            bbox = bbox.tolist()
            x1, y1, x2, y2, score, classID = bbox
            if detector.classes[classID] in classes and score >= threshold:
                dlist.append({
                        'class': detector.classes[classID],
                        'score': float(score),
                        'left': int(x1),
                        'right': int(x2),
                        'top': int(y1),
                        'bottom': int(y2),
                    })
        detections.append(dlist)

    detections = [nms(detection, 0.1) for detection in detections]

    print('json'+json.dumps(detections), flush=True)