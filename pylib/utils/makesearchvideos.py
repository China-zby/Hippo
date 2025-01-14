import os
import cv2
import json
import pickle
import argparse
from alive_progress import alive_bar

parse = argparse.ArgumentParser()
parse.add_argument('--data_dir', dest='data_dir', type=str, default='', required=True)
parse.add_argument('--data_name', dest='data_name', type=str, default='', required=True)
parse.add_argument('--data_type', dest='data_type', type=str, default='', required=True)
parse.add_argument('--cut_video', dest='cut_video', type=int, default=10, required=True)
args = parse.parse_args()

cut_video = args.cut_video
data_root = os.path.join(args.data_dir, "dataset", args.data_name, args.data_type)
save_data_root = os.path.join(args.data_dir, "dataset", args.data_name, "search")

if not os.path.exists(save_data_root):
    os.makedirs(save_data_root)
    
video_data_root = os.path.join(data_root, "video")
video_save_data_root = os.path.join(save_data_root, "video")

tracks_data_root = os.path.join(data_root, "tracks")
tracks_save_data_root = os.path.join(save_data_root, "tracks")

if not os.path.exists(video_save_data_root) or not os.path.exists(tracks_save_data_root):
    os.makedirs(video_save_data_root, exist_ok=True)
    os.makedirs(tracks_save_data_root, exist_ok=True)

print(tracks_save_data_root)

video_list = os.listdir(video_data_root)
with alive_bar(len(video_list)) as bar:
    for video_name in video_list:
        video_id = video_name.split('.')[0]
        video_capture = cv2.VideoCapture(os.path.join(video_data_root, video_name))
        
        # Get video properties
        frame_rate = video_capture.get(cv2.CAP_PROP_FPS)
        frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_number = frame_rate * cut_video
        
        # Define the codec and create VideoWriter object
        video_writer = cv2.VideoWriter(os.path.join(video_save_data_root, video_name), cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (frame_width, frame_height))
        
        frame_idx = 0
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break
            if frame_idx == frame_number:
                break
            video_writer.write(frame)
            frame_idx += 1
        
        video_capture.release()
        video_writer.release()
        
        # Copy tracks
        track_txt_name = video_id + '.txt'
        track_pkl_name = video_id + '.pkl'
        track_json_name = video_id + '.json'
        
        txt_data = open(os.path.join(tracks_data_root, track_txt_name), 'r').readlines()
        pkl_data = pickle.load(open(os.path.join(tracks_data_root, track_pkl_name), 'rb'))
        json_data = json.load(open(os.path.join(tracks_data_root, track_json_name), 'r'))
        
        save_txt_path = os.path.join(tracks_save_data_root, track_txt_name)
        save_pkl_path = os.path.join(tracks_save_data_root, track_pkl_name)
        save_json_path = os.path.join(tracks_save_data_root, track_json_name)
        
        # print("video_id: {}, frame_number: {}, frame_idx: {}".format(video_id, frame_number, frame_idx))
        if not os.path.exists(save_txt_path):
            txt_writer = open(save_txt_path, 'w')
            for txt_line in txt_data:
                txt_list = txt_line.strip().split(',')
                if int(txt_list[0]) > frame_number - 1:
                    break
                txt_writer.write(txt_line)
        
        if not os.path.exists(save_pkl_path):
            track_id_remap = {}
            new_pkl_data = {}
            for track_id in pkl_data.keys():
                frame_bound, track = pkl_data[track_id]
                if frame_bound[0] > frame_number - 1:
                    continue
                frame_bound[1] = min(frame_bound[1], frame_number - 1)
                new_track = []
                for x1, y1, x2, y2, frame_idx, classid in track:
                    if frame_idx > frame_number - 1:
                        break
                    new_track.append([x1, y1, x2, y2, frame_idx, classid])
                
                if track_id not in track_id_remap:
                    track_id_remap[track_id] = len(track_id_remap)
                new_pkl_data[track_id_remap[track_id]] = [frame_bound, new_track]
            pickle.dump(new_pkl_data, open(save_pkl_path, 'wb'))
        
        if not os.path.exists(save_json_path):
            new_dets = []
            for frame_idx, dets in enumerate(json_data):
                if frame_idx > frame_number - 1:
                    break
                new_dets.append(dets)
            json.dump(new_dets, open(save_json_path, 'w'))
        
        bar()