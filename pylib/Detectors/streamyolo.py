import os
import cv2
import sys
import json
import time
import torch
import numpy as np
from dataloader import read_im
from torchvision.ops import batched_nms
from mmcv.runner import load_checkpoint

from mmdet.apis import inference_detector, show_result_pyplot
from mmdet.models import build_detector
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

os.environ["CUDA_VISIBLE_DEVICES"]="1"

data_root = sys.argv[1]
batch_size = int(sys.argv[2])
param_width = int(sys.argv[3])
param_height = int(sys.argv[4])
threshold = float(sys.argv[5])
classes = sys.argv[6]
label = sys.argv[7]
modelsize = sys.argv[8]

# modelsize = "l"
# threshold = 0.25
# classes = "car,bus,truck"

classes = classes.split(",")
classIDs = {2: "car",
            5: "bus",
            7: "truck"}

if modelsize.lower() == "s":
    config = "/home/lzp/MyGreatVoyage/YOLV/StreamYOLO/cfgs/s_s50_onex_dfp_tal_flip.py"
    weights = "./weights/Detectors/general/s_s50_one_x.pth"
elif modelsize.lower() == "m":
    config = "/home/lzp/MyGreatVoyage/YOLV/StreamYOLO/cfgs/m_s50_onex_dfp_tal_flip.py"
    weights = "./weights/Detectors/general/m_s50_one_x.pth"
elif modelsize.lower() == "l":
    config = "/home/lzp/MyGreatVoyage/YOLV/StreamYOLO/cfgs/l_s50_onex_dfp_tal_filp.py"
    weights = "./weights/Detectors/general/l_s50_one_x.pth"

sys.path.pop(-1)
# sys.path.append(os.path.join(data_root, 'yolov5/'))
sys.path.append('/home/xyc/otif/otif-dataset/StreamYOLO')

from yolox.exp import get_exp

def time_synchronized():
    """pytorch-accurate time"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

def preproc(img, input_size, swap=(2, 0, 1)):
    resized_img = cv2.resize(img, (input_size[1], input_size[0]), interpolation=cv2.INTER_LINEAR,)
    resized_img = resized_img.transpose(swap)
    return resized_img

def inference(outputs, conf_thre=0.01, nms_thresh=0.65, in_scale=1.0):
    box_corner = outputs.new(outputs.shape)
    box_corner[:, 0] = outputs[:, 0] - outputs[:, 2] / 2
    box_corner[:, 1] = outputs[:, 1] - outputs[:, 3] / 2
    box_corner[:, 2] = outputs[:, 0] + outputs[:, 2] / 2
    box_corner[:, 3] = outputs[:, 1] + outputs[:, 3] / 2
    outputs[:, :4] = box_corner[:, :4]

    class_conf, class_pred = torch.max(outputs[:, 5:], 1, keepdim=True)
    conf_mask = (outputs[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
    detections = torch.cat((outputs[:, :5], class_conf, class_pred.float()), 1)
    detections = detections[conf_mask]

    nms_out_index = batched_nms(
        detections[:, :4],
        detections[:, 4] * detections[:, 5],
        detections[:, 6],
        nms_thresh,
    )

    detections = detections[nms_out_index].cpu().detach().numpy()
    return detections[:, :4] / in_scale, detections[:, 4] * detections[:, 5], detections[:, 6].astype(np.int32), None

exp = get_exp(config, None)
model = exp.get_model()
model.cuda()
model.eval()
ckpt = torch.load(weights, map_location="cpu")
model.load_state_dict(ckpt["model"])
print("loaded checkpoint done.")
# model = fuse_model(model)
model.eval()
model.half()
# tensor_type = torch.cuda.FloatTensor
tensor_type = torch.cuda.HalfTensor

# warm up the GPU
img = cv2.imread("enhanceimage.jpeg")
w_img, h_img = img.shape[1], img.shape[0]
tmp_image = torch.ones(1, 3, int(h_img), int(w_img)).type(tensor_type)
_buffer = None
for i in range(10):
    _, _ = model(tmp_image, buffer=_buffer, mode='on_pipe')

torch.cuda.synchronize()

buffer = None

stdin = sys.stdin.detach()
while True:
    imgs = read_im(stdin)
    if imgs is None:
        break
    frames = []
    for im0 in imgs:
        h_img, w_img = im0.shape[0], im0.shape[1]
        frame = preproc(im0, input_size=(h_img, w_img))  # [3,600,960]
        frames.append(frame)
    with torch.no_grad():
        frames = torch.from_numpy(frames).type(tensor_type)    # [1,3,600,960]
        results, buffer = model(frames, buffer=buffer, mode='on_pipe')
    
    detections = []
    for result in results:    
        bboxes, scores, labels, masks = inference(result[0])
        
        dlist = []
        # print(bboxes, scores, labels, masks)
        for box, score, label in zip(bboxes, scores, labels):
            if score > threshold and label in list(classIDs.keys()):
                dlist.append({"left": int(box[0]), 
                              "top": int(box[1]),
                              "right": int(box[2]), 
                              "bottom": int(box[3]),
                              "score": float(score),
                              "class": classIDs[label]})
        detections.append(dlist)
        #         print(box, score, label)
        #         cv2.rectangle(im0, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
        #         cv2.putText(im0, classIDs[label], (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # cv2.imwrite("result.jpeg", im0)
    
    detections = [nms(detection, 0.1) for detection in detections]
    
    print('json'+json.dumps(detections), flush=True)