import os
import json
import random
import pickle
import argparse 

from alive_progress import alive_bar

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', dest='data_dir', type=str, default='', required=True)
parser.add_argument('--data_name', dest='data_name', type=str, default='', required=True)
parser.add_argument('--data_type', dest='data_type', type=str, default='', required=True)
parser.add_argument('--cut_number', dest='cut_number', type=int, default=10, required=True)
args = parser.parse_args()

data_root = os.path.join(args.data_dir, "dataset", args.data_name, args.data_type)
save_data_root = os.path.join(args.data_dir, "dataset", args.data_name, "streamline")

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

camera_data = []
for video_name in video_list:
    video_id = video_name.split('.')[0]
    
    track_txt_name = video_id + '.txt'
    track_pkl_name = video_id + '.pkl'
    track_json_name = video_id + '.json'
    
    txt_data = open(os.path.join(tracks_data_root, track_txt_name), 'r').readlines()
    pkl_data = pickle.load(open(os.path.join(tracks_data_root, track_pkl_name), 'rb'))
    json_data = json.load(open(os.path.join(tracks_data_root, track_json_name), 'r'))
    
    camera_data.append([video_id, video_name, len(pkl_data.keys()), 
                        os.path.join(tracks_data_root, track_txt_name), 
                        os.path.join(tracks_data_root, track_pkl_name),
                        os.path.join(tracks_data_root, track_json_name),
                        os.path.join(video_data_root, video_name)])

pos_num = int(args.cut_number / 3)
med_num = int(args.cut_number / 3)
neg_num = args.cut_number - pos_num - med_num
video_num = len(camera_data)
video_pos_num = int(video_num / 3)
video_med_num = int(video_num / 3)
video_neg_num = video_num - video_pos_num - video_med_num
camera_data = sorted(camera_data, key=lambda x: x[2], reverse=True)
chosen_camera_data = random.sample(camera_data[:video_pos_num], pos_num) + \
                        random.sample(camera_data[video_pos_num:video_pos_num+video_med_num], med_num) + \
                           random.sample(camera_data[video_pos_num+video_med_num:], neg_num)
                           
print([x[2] for x in chosen_camera_data])

with alive_bar(len(chosen_camera_data)) as bar:               
    for new_video_id, \
        (video_id,\
        video_name,\
        track_number,\
        txt_path, pkl_path,\
        json_path, video_path) in enumerate(chosen_camera_data):  
  
        save_video_path = os.path.join(video_save_data_root, str(new_video_id) + '.mp4')
        save_txt_path = os.path.join(tracks_save_data_root, str(new_video_id) + '.txt')
        save_pkl_path = os.path.join(tracks_save_data_root, str(new_video_id) + '.pkl')
        save_json_path = os.path.join(tracks_save_data_root, str(new_video_id) + '.json')
        
        os.system('cp {} {}'.format(video_path, save_video_path))
        os.system('cp {} {}'.format(txt_path, save_txt_path))
        os.system('cp {} {}'.format(pkl_path, save_pkl_path))
        os.system('cp {} {}'.format(json_path, save_json_path))
        bar()