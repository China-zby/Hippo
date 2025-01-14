import os
import cv2
import json
import random
import argparse
from tqdm import tqdm
from alive_progress import alive_bar

parse = argparse.ArgumentParser(description='Make crops')
parse.add_argument("--data_root", type=str, default="./")
parse.add_argument("--data_name", type=str, default="amsterdam")
parse.add_argument("--data_type", type=str, default="tracker")
args = parse.parse_args()

classesMap = {"car": 2, "bus": 5, "truck": 7}
classesMapReverse = {2: "car", 5: "bus", 7: "truck"}

data_dir = os.path.join(args.data_root, args.data_name, args.data_type)
train_videoids, test_videoids = [], []
video_id_list = [video_path_name.split(".")[0] for video_path_name in os.listdir(os.path.join(data_dir, "video"))]
random.shuffle(video_id_list)
train_videoids = video_id_list[:int(len(video_id_list) * 0.8)]
test_videoids = video_id_list[int(len(video_id_list) * 0.8):]

video_id_list = os.listdir(os.path.join(data_dir, "video"))
video_id_list.sort()

save_dir = os.path.join(data_dir, "crops/images")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

frame_gap = 16
max_tracks = 1000   

train_image_list, test_image_list, query_image_list = [], [], []

with alive_bar(len(video_id_list)) as bar:
    trackid_remap_dict = {}
    for video_path_name in video_id_list:
        videoid = video_path_name.split(".")[0]
        video_path = os.path.join(data_dir, "video", f"{videoid}.mp4")
        track_path = os.path.join(data_dir, "tracks", f"{videoid}.json")
        
        track_data = {}
        track_before_filter_data = {}
        with open(track_path, "r") as f:
            # track_list = f.readlines()
            # for frame_line in tqdm(track_list):
            #     frameid, trackid, x, y, w, h, conf, clsid, _, _, _ = frame_line.split(",")
            #     frameid, trackid, x, y, w, h, conf, clsid = int(frameid), int(trackid), float(x), float(y), float(w), float(h), float(conf), int(float(clsid))
            #     if trackid not in track_before_filter_data:
            #         track_before_filter_data[trackid] = {"frame_bound": {"start_frame_id": frameid,
            #                                                              "end_frame_id": frameid}, 
            #                                              "position_list": [[x + w/2, y + h/2]],
            #                                              "data": [[frameid, trackid, x, y, w, h, conf, clsid]]}
            #     else:
            #         track_before_filter_data[trackid]["frame_bound"]["end_frame_id"] = frameid
            #         track_before_filter_data[trackid]["position_list"].append([x + w/2, y + h/2])
            #         track_before_filter_data[trackid]["data"].append([frameid, trackid, x, y, w, h, conf, clsid])
            gt_data = json.load(open(track_path, "r"))
            for frameid, trackdata in enumerate(gt_data):
                frameid += 1
                if trackdata == [] or trackdata is None: continue
                for objectdata in trackdata:
                    x1, y1, x2, y2 = objectdata["left"], objectdata["top"],\
                                        objectdata["right"], objectdata["bottom"]
                    trackid = int(objectdata["track_id"])
                    conf = float(objectdata["score"])
                    clsid = classesMap[objectdata["class"]]
                    if trackid not in track_before_filter_data:
                        track_before_filter_data[trackid] = {"frame_bound": {"start_frame_id": frameid,
                                                                            "end_frame_id": frameid}, 
                                                            "position_list": [[(x1 + x2)/2, (y1 + y2)/2]],
                                                            "data": [[frameid, trackid, x1, y1, x2 - x1, y2 - y1, conf, clsid]]}
                    else:
                        track_before_filter_data[trackid]["frame_bound"]["end_frame_id"] = frameid
                        track_before_filter_data[trackid]["position_list"].append([(x1 + x2)/2, (y1 + y2)/2])
                        track_before_filter_data[trackid]["data"].append([frameid, trackid, x1, y1, x2 - x1, y2 - y1, conf, clsid])
                        
            all_trackids = set()
            for trackid in track_before_filter_data.keys():
                if track_before_filter_data[trackid]["frame_bound"]["end_frame_id"] - track_before_filter_data[trackid]["frame_bound"]["start_frame_id"] < 60:
                    continue
                sx, sy = track_before_filter_data[trackid]["position_list"][0]
                ex, ey = track_before_filter_data[trackid]["position_list"][-1]
                if ((ex - sx) ** 2 + (ey - sy) ** 2)**0.5 < 100:
                    continue
                for dataline in track_before_filter_data[trackid]["data"]:
                    frameid, trackid, x, y, w, h, conf, clsid = dataline
                    trackinfo = f"{videoid}_{trackid}"
                    if trackinfo not in trackid_remap_dict:
                        trackid_remap_dict[trackinfo] = len(trackid_remap_dict) + 1
                    remap_frameid = frameid + 1
                    remap_trackid = trackid_remap_dict[trackinfo]
                    if remap_frameid not in track_data:
                        track_data[remap_frameid] = []
                    all_trackids.add(remap_trackid)
                    track_data[remap_frameid].append([remap_trackid, x, y, w, h, conf, clsid])
        
        video_capture = cv2.VideoCapture(video_path)
        width, height = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        train_trackids = random.sample(all_trackids, int(len(all_trackids) * 0.8))
        
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break
            frameid = int(video_capture.get(cv2.CAP_PROP_POS_FRAMES))
            if frameid not in track_data or frameid % frame_gap != 0:
                continue
            detections = track_data[frameid]
            for detection in detections:
                trackid = detection[0]
                x1, y1, w, h = detection[1], detection[2], detection[3], detection[4]
                x2, y2 = x1 + w, y1 + h
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
                if x2 - x1 < 10 or y2 - y1 < 10: continue
                crop_image = frame[y1:y2, x1:x2]
                # print(crop_image.shape)
                if trackid in train_trackids: 
                    train_image_list.append(f"images/{trackid:06d}_{frameid:04d}.jpg")
                else: 
                    randomid = random.randint(0, 100)
                    if randomid < 90:
                        test_image_list.append(f"images/{trackid:06d}_{frameid:04d}.jpg")
                    else:
                        query_image_list.append(f"images/{trackid:06d}_{frameid:04d}.jpg")
                cv2.imwrite(os.path.join(save_dir, f"{trackid:06d}_{frameid:04d}.jpg"), crop_image)
        bar()

# build image train/test/query list 
train_list_path, test_list_path = os.path.join(data_dir, "crops", "train.txt"), os.path.join(data_dir, "crops", "test.txt")
query_list_path = os.path.join(data_dir, "crops", "query.txt")
with open(train_list_path, "w") as f:
    for image_info in train_image_list:
        f.write(f"{image_info}\n")

with open(test_list_path, "w") as f:
    for image_info in test_image_list:
        f.write(f"{image_info}\n")
        
with open(query_list_path, "w") as f:
    for image_info in query_image_list:
        f.write(f"{image_info}\n")