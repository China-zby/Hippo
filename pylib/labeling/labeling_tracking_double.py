import os
import cv2
import sys
import json
import torch
import pickle
import argparse
import numpy as np
from alive_progress import alive_bar
from tracker.byte_tracker import BYTETracker

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

def get_good_tracklets(detections, min_length=30): # filter out short tracklets
    tracklets = {}
    trackiddict = {}
    for i, detection in enumerate(detections):
        if detection is None: continue
        for det in detection:
            if det['track_id'] not in tracklets:
                tracklets[det['track_id']] = []
            tracklets[det['track_id']].append([i + 1, det])
    good_tracklets = {}
    for track_id, tracklet in tracklets.items():
        if len(tracklet) >= min_length or tracklet[-1][0] == len(detections):
            good_tracklets[track_id] = tracklet 
    good_detections = []
    for i, detection in enumerate(detections):
        if detection is None:
            good_detections.append(None)
        else:
            frame_detection = []
            for det in detection:
                if det['track_id'] in good_tracklets:
                    if det['track_id'] not in trackiddict:
                        trackiddict[det['track_id']] = len(trackiddict) + 1
                    det['track_id'] = trackiddict[det['track_id']]
                    frame_detection.append(det)
            good_detections.append(frame_detection) # [det for det in detection if det['track_id'] in good_tracklets]
    return good_detections

def save_txt(save_path, detections):
    with open(save_path, 'w') as f:
        for frame_id, detection in enumerate(detections):
            if detection is None: continue
            for det in detection:
                tid = det['track_id']
                x1, y1, x2, y2 = det['left'], det['top'], det['right'], det['bottom']
                w, h = x2 - x1, y2 - y1
                score = det['score']
                classID = name_inverse_dict2[det['class']]
                f.write(f"{frame_id},{tid},{x1},{y1},{w},{h},{score:.4f},{classID},-1,-1,-1\n")

def save_pkl(save_path, detections):
    query_result = {}
    for frame_id, detection in enumerate(detections):
        if detection is None: continue
        for det in detection:
            tid = det['track_id']
            x1, y1, x2, y2 = det['left'], det['top'], det['right'], det['bottom']
            w, h = x2 - x1, y2 - y1
            score = det['score']
            classID = name_inverse_dict2[det['class']]
            if tid not in query_result:
                query_result[tid] = [[frame_id, frame_id], [[x1, y1, x1+w, y1+h, frame_id, classID]]]
            else:
                if frame_id < query_result[tid][0][0]:
                    query_result[tid][0][0] = frame_id
                if frame_id > query_result[tid][0][1]:
                    query_result[tid][0][1] = frame_id
                query_result[tid][1].append([x1, y1, x1+w, y1+h, frame_id, classID])
    pickle.dump(query_result, open(save_path, 'wb'))

def save_json(save_path, detections):
    json.dump(detections, open(save_path, 'w'))

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default='/home/xyc/otif/otif-dataset/dataset')
parser.add_argument('--data_name', type=str, default='taipei')
parser.add_argument('--data_flag', type=str, default='train/video')
parser.add_argument('--save_flag', type=str, default='train/tracks')
parser.add_argument('--width', type=int, default=640)
parser.add_argument('--height', type=int, default=640)
parser.add_argument('--threshold', type=float, default=0.1)
parser.add_argument('--device', type=str, default='0')
parser.add_argument('--min_video_length', type=int, default=30)
parser.add_argument('--min_move_length', type=int, default=30)
parser.add_argument('--min_area', type=int, default=100)
parser.add_argument('--save_txt', action='store_true', default=False)
parser.add_argument('--save_pkl', action='store_true', default=False)
parser.add_argument('--save_json', action='store_true', default=False)

parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
parser.add_argument("--match_thresh", type=float, default=0.9, help="matching threshold for tracking")
parser.add_argument("--match_thresh_second", type=float, default=0.7, help="matching threshold for tracking")
parser.add_argument("--match_thresh_unconfirm", type=float, default=0.9, help="matching threshold for tracking")
parser.add_argument("--min-box-area", type=float, default=100, help='filter out tiny boxes')
parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")

args = parser.parse_args()

param_width, param_height = args.width, args.height
threshold = args.threshold

sys.path.append(os.path.join(args.data_root, 'yolov5'))
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

try:
    mask_json_path = os.path.join(args.data_root, 'dataset', args.data_name, 'background.json')
    mask_image_path = os.path.join(args.data_root, 'dataset', args.data_name, 'background.jpg')
    mask_data = json.load(open(mask_json_path, 'r'))
    background_image = cv2.imread(mask_image_path)
    mask_image = np.zeros(background_image.shape[:2], dtype=np.uint8)

    for area in mask_data:
        area_mask = area['rectMask'] # 'xMin': 506, 'yMin': 216.5, 'width': 28, 'height'
        x1, y1, x2, y2 = int(area_mask['xMin']), \
                        int(area_mask['yMin']), \
                        int(area_mask['xMin']+area_mask['width']), \
                        int(area_mask['yMin']+area_mask['height'])
        mask_image[y1:y2, x1:x2] = 255

    cv2.imwrite(os.path.join(args.data_root, 'dataset', args.data_name, 'mask.jpg'), mask_image)
except:
    print('no mask')
    mask_image = None

device = torch.device('cuda:0')
model1 = DetectMultiBackend(weight_path1,
                            device=device,
                            dnn=dnn, 
                            data=data_path1, 
                            fp16=half)

device = torch.device('cuda:1')
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

videoList = [os.path.join(args.data_root, "dataset", args.data_name, args.data_flag, path_name) for path_name in os.listdir(os.path.join(args.data_root, "dataset", args.data_name, args.data_flag)) if path_name.endswith('.mp4')]
dataset = LoadImages(videoList, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
save_tracklet_dir = os.path.join(args.data_root, "dataset", args.data_name, args.save_flag)
if not os.path.exists(save_tracklet_dir):
    os.makedirs(save_tracklet_dir)

im0list = []
last_videoname = -1

with alive_bar(len(dataset)) as bar:
    for path, im, im0s, vid_cap, s in dataset:
        pathID = int(path.split('/')[-1].split('.')[0])
        if os.path.exists(os.path.join(save_tracklet_dir, f'{pathID}.json')) and \
            os.path.exists(os.path.join(save_tracklet_dir, f'{pathID}.txt')) and \
                os.path.exists(os.path.join(save_tracklet_dir, f'{pathID}.pkl')):
            continue
        if pathID != last_videoname:
            if last_videoname != -1:
                good_tracklets = get_good_tracklets(tracklets, args.min_video_length)
                save_flag = args.save_flag.split("/")[0]
                videoWriter = cv2.VideoWriter(f'./labelingTest/{args.data_name}_{save_flag}_{last_videoname}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (im0s.shape[1], im0s.shape[0]))
                for im0, frame_detection in zip(im0list, good_tracklets):
                    if frame_detection is not None:
                        for detection in frame_detection:
                            track_id = detection['track_id']
                            x1, y1, x2, y2 = detection['left'], detection['top'], detection['right'], detection['bottom']
                            cv2.rectangle(im0, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                            cv2.putText(im0, str(track_id), (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                    videoWriter.write(im0)
                videoWriter.release()
                
                if args.save_json: save_json(os.path.join(save_tracklet_dir, f'{last_videoname}.json'), good_tracklets)
                if args.save_txt: save_txt(os.path.join(save_tracklet_dir, f'{last_videoname}.txt'), good_tracklets)
                if args.save_pkl: save_pkl(os.path.join(save_tracklet_dir, f'{last_videoname}.pkl'), good_tracklets)
                bar()
            
            im0list = []
            tracklets = []
            tracker = BYTETracker(args)
            last_videoname = pathID
        
        im2 = torch.from_numpy(im).to(model2.device)
        im2 = im2.half() if model2.fp16 else im2.float()  # uint8 to fp16/32
        im2 /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im2.shape) == 3:
            im2 = im2[None]
        
        im = torch.from_numpy(im).to(model1.device)
        im = im.half() if model1.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        # Inference
        with torch.no_grad():
            pred1 = model1(im, augment=augment, visualize=visualize)
            pred2 = model2(im2, augment=augment, visualize=visualize)

        # NMS
        pred1 = non_max_suppression(pred1, conf_thres, iou_thres, classes=classes1, agnostic=agnostic_nms, max_det=max_det)
        pred2 = non_max_suppression(pred2, conf_thres, iou_thres, classes=classes2, agnostic=agnostic_nms, max_det=max_det)
        
        detections = []
        im0 = im0s.copy()
        for det1, det2 in zip(pred1, pred2):  # per image
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
                
                for *x1y1x2y2, conf, cls in reversed(det):
                    w, h = x1y1x2y2[2] - x1y1x2y2[0], x1y1x2y2[3] - x1y1x2y2[1]
                    if w * h < args.min_area:
                        continue
                    xc, yc = x1y1x2y2[0] + w / 2, x1y1x2y2[1] + h / 2
                    if mask_image is not None:
                        if mask_image[int(yc), int(xc)] == 255:
                            continue
                    detections.append([int(x1y1x2y2[0]), int(x1y1x2y2[1]), int(x1y1x2y2[2]), int(x1y1x2y2[3]), float(conf), int(cls)])
        detections = np.array(detections)
        output_stracks = tracker.update(detections)
        
        frame_tracklets = []
        for strack in output_stracks: # track_id
            bbox = strack.tlbr
            frame_tracklets.append({"left": int(bbox[0]), "top": int(bbox[1]), "right": int(bbox[2]), "bottom": int(bbox[3]), "track_id": strack.track_id, "score": strack.score, "class": name_dict2[strack.classID]})

        im0list.append(im0)
        if len(frame_tracklets) > 0: tracklets.append(frame_tracklets)
        else: tracklets.append(None)
        
    good_tracklets = get_good_tracklets(tracklets, args.min_video_length)
    save_flag = args.save_flag.split("/")[0]
    videoWriter = cv2.VideoWriter(f'./labelingTest/{args.data_name}_{save_flag}_{last_videoname}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (im0s.shape[1], im0s.shape[0]))
    for im0, frame_detection in zip(im0list, good_tracklets):
        if frame_detection is not None:
            for detection in frame_detection:
                track_id = detection['track_id']
                x1, y1, x2, y2 = detection['left'], detection['top'], detection['right'], detection['bottom']
                cv2.rectangle(im0, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(im0, str(track_id), (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        videoWriter.write(im0)
    videoWriter.release()
    if args.save_json: save_json(os.path.join(save_tracklet_dir, f'{last_videoname}.json'), good_tracklets)
    if args.save_txt: save_txt(os.path.join(save_tracklet_dir, f'{last_videoname}.txt'), good_tracklets)
    if args.save_pkl: save_pkl(os.path.join(save_tracklet_dir, f'{last_videoname}.pkl'), good_tracklets)
    bar()