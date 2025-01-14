
from lib2to3.pytree import Base

import os
import cv2
import sys
import json
import numpy
import shutil
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

os.chdir(os.path.join(data_root, 'darknet-alexey/'))
sys.path.append('./')
import darknet

def eprint(s):
    sys.stderr.write(str(s) + "\n")
    sys.stderr.flush()

if classes != '':
    classes = {cls.strip(): True for cls in classes.split(',')}
else:
    classes = None

detector_label = "generic" # label
if detector_label.startswith('caldot'):
    detector_label = 'caldot'
if detector_label in ['amsterdam', 'jackson', 'taipei', 'camera_envs']:
    detector_label = 'generic'

config_path = os.path.join(data_root, 'yolov3', detector_label, 'yolov3-{}x{}-test.cfg'.format(param_width, param_height))
meta_path = os.path.join(data_root, 'yolov3', detector_label, 'obj.data')
names_path = os.path.join(data_root, 'yolov3', detector_label, 'obj.names')

if detector_label == 'generic':
    weight_path = os.path.join(data_root, 'yolov3', detector_label, 'yolov3.best')
else:
    weight_path = os.path.join(data_root, 'yolov3', label, 'yolov3-{}x{}.best'.format(param_width, param_height))

with open(config_path, 'r') as f:
    tmp_config_buf = ''
    for line in f.readlines():
        line = line.strip()
        if line.startswith('width='):
            line = 'width={}'.format(param_width)
        if line.startswith('height='):
            line = 'height={}'.format(param_height)
        tmp_config_buf += line + "\n"
tmp_config_path = '/tmp/yolov3-{}.cfg'.format(os.getpid())
with open(tmp_config_path, 'w') as f:
    f.write(tmp_config_buf)

# Write out our own obj.data which has direct path to obj.names.
tmp_obj_names = '/tmp/obj-{}.names'.format(os.getpid())
shutil.copy(names_path, tmp_obj_names)

with open(meta_path, 'r') as f:
    tmp_meta_buf = ''
    for line in f.readlines():
        line = line.strip()
        if line.startswith('names='):
            line = 'names={}'.format(tmp_obj_names)
        tmp_meta_buf += line + "\n"
tmp_obj_meta = '/tmp/obj-{}.data'.format(os.getpid())
with open(tmp_obj_meta, 'w') as f:
    f.write(tmp_meta_buf)

# print(tmp_obj_meta, file=open("/root/autodl-tmp/otifpipeline/log.txt", "a"))
# Finally we can load YOLOv3.
net, class_names, _ = darknet.load_network(tmp_config_path, tmp_obj_meta, weight_path, batch_size=batch_size)
# print('Loaded YOLOv3 model with {} classes.'.format(len(class_names)), file=open("/root/autodl-tmp/otifpipeline/log.txt", "a"))
os.remove(tmp_config_path)
os.remove(tmp_obj_names)
os.remove(tmp_obj_meta)

def image_preporcess(image, target_size, gt_boxes=None):
    ih, iw    = target_size # resize 尺寸
    h,  w, _  = image.shape # 原始图片尺寸

    scale = min(iw/w, ih/h)
    nw, nh  = int(scale * w), int(scale * h) # 计算缩放后图片尺寸
    image_resized = cv2.resize(image, (nw, nh))
    # 制作一张画布，画布的尺寸就是我们想要的尺寸
    image_paded = numpy.full(shape=[ih, iw, 3], fill_value=114)
    dw, dh = (iw - nw) // 2, (ih-nh) // 2
    # 将缩放后的图片放在画布中央
    image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized
    # image_paded = image_paded / 255.

    if gt_boxes is None:
        return image_paded, scale, dw, dh

    else:   # 训练网络时需要对 groudtruth box 进行矫正
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
        return image_paded, gt_boxes, scale, dw, dh
seen = 0
stdin = sys.stdin.detach()
while True:
    window_ims = read_im(stdin)
    if window_ims is None:
        break
        
    im = []
    for window_im in window_ims:
        if window_im.shape[0] != param_height or window_im.shape[1] != param_width:
            window_im, scale, dw, dh = image_preporcess(window_im, [param_height, param_width]) # cv2.resize(window_im, (param_width, param_height))
        else:
            scale, dw, dh = 1.0, 0, 0
        im.append(window_im)
    im = numpy.array(im)

    height, width = im.shape[1], im.shape[2]
    # print('height: {}, width: {}'.format(height, width), file=open("/root/autodl-tmp/otifpipeline/log.txt", "a"))
    arr = im.transpose((0, 3, 1, 2))
    arr = numpy.ascontiguousarray(arr.flat, dtype='float32')/255.0
    darknet_images = arr.ctypes.data_as(darknet.POINTER(darknet.c_float))
    darknet_images = darknet.IMAGE(width, height, 3, darknet_images)
    raw_detections = darknet.network_predict_batch(net, darknet_images, batch_size, width, height, threshold, 0.5, None, 0, 0)
    detections = []
    for idx in range(batch_size):
        num = raw_detections[idx].num
        raw_dlist = raw_detections[idx].dets
        darknet.do_nms_obj(raw_dlist, num, len(class_names), 0.45)
        raw_dlist = darknet.remove_negatives(raw_dlist, class_names, num)
        dlist = []
        for cls, score, (cx, cy, w, h) in raw_dlist:
            
            x1, y1, x2, y2 = int(cx-w/2), int(cy-h/2), int(cx+w/2), int(cy+h/2)
            x1, y1, x2, y2 = int((x1 - dw)/scale), int((y1 - dh)/scale), int((x2 - dw)/scale), int((y2 - dh)/scale)
            w, h = x2 - x1, y2 - y1
            
            if cls not in classes.keys() or w < 0 or h < 0 or w > width or h > height:
                continue
            
            dlist.append({
                'class': cls,
                'score': float(score),
                'left': x1,
                'right': x2,
                'top': y1,
                'bottom': y2,
            })
            # print('dlist: {}'.format(dlist), file=open("/home/lzp/go-work/src/otifpipeline/log2.txt", "a"))
        detections.append(dlist)

    detections = [nms(detection, 0.1) for detection in detections]        

    darknet.free_batch_detections(raw_detections, batch_size)
    print('json'+json.dumps(detections), flush=True)
    seen += batch_size