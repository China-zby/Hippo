import cv2
import sys
import json

import numpy as np
from dataloader import read_im
from mmcv.transforms import Compose
from mmdet.apis import init_detector, inference_detector
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

classes = classes.split(",")

if modelsize.lower() == "r50100":
    config_file = "./pylib/Detectors/configs/sparse_rcnn/sparse-rcnn_r50_fpn_ms-480-800-3x_coco.py"
    checkpoint_file = "./weights/Detectors/general/sparse_rcnn_r50_fpn_mstrain_480-800_3x_coco_20201218_154234-7bc5c054.pth"
elif modelsize.lower() == "r50300":
    config_file = "./pylib/Detectors/configs/sparse_rcnn/sparse-rcnn_r50_fpn_300-proposals_crop-ms-480-800-3x_coco.py"
    checkpoint_file = "./weights/Detectors/general/sparse_rcnn_r50_fpn_300_proposals_crop_mstrain_480-800_3x_coco_20201223_024605-9fe92701.pth"
elif modelsize.lower() == "r101100":
    config_file = "./pylib/Detectors/configs/sparse_rcnn/sparse-rcnn_r101_fpn_ms-480-800-3x_coco.py"
    checkpoint_file = "./weights/Detectors/general/sparse_rcnn_r101_fpn_mstrain_480-800_3x_coco_20201223_121552-6c46c9d6.pth"
elif modelsize.lower() == "r101300":
    config_file = "./pylib/Detectors/configs/sparse_rcnn/sparse-rcnn_r101_fpn_300-proposals_crop-ms-480-800-3x_coco.py"
    checkpoint_file = "./weights/Detectors/general/sparse_rcnn_r101_fpn_300_proposals_crop_mstrain_480-800_3x_coco_20201223_023452-c23c3564.pth"

device=f'cuda:{device_id}'

# Build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device=device)
model.test_cfg = dict(
    rpn=dict(nms_pre=1000,
    min_bbox_size=0,
    nms=dict(type='soft_nms', iou_threshold=0.4, agnostic=True),
    max_per_img=200),
    rcnn=dict(score_thr=threshold,
    nms=dict(type='soft_nms', iou_threshold=0.4, agnostic=True),
    max_per_img=200)
)

# Build test pipeline
model.cfg.test_dataloader.dataset.pipeline[0].type = 'LoadImageFromNDArray'
test_pipeline = Compose(model.cfg.test_dataloader.dataset.pipeline)

stdin = sys.stdin.detach()
while True:
    imgs = read_im(stdin)
    if imgs is None:
        break
    ims = []
    for im in imgs: 
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        ims.append(im)
    results = inference_detector(model, ims, test_pipeline=test_pipeline)

    detections = []
    for result in results:
        dlist = []
        for classID, bbox, score in zip(result.pred_instances.labels, result.pred_instances.bboxes, result.pred_instances.scores):
            if model.dataset_meta['classes'][classID] not in classes: continue
            x1, y1, x2, y2 = bbox
            if float(score) < threshold: continue
            dlist.append({
                'class': model.dataset_meta['classes'][classID],
                'score': float(score),
                'left': int(x1),
                'right': int(x2),
                'top': int(y1),
                'bottom': int(y2),
            })
        detections.append(dlist)
        
    detections = [nms(detection, 0.1) for detection in detections]

    print('json'+json.dumps(detections), flush=True)