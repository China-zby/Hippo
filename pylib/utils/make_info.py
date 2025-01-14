import os
import cv2
import subprocess
from tqdm import tqdm

env_flag = "one_second"
suffixs = ["json"]
dataset_root = "/home/lzp/otif-dataset/dataset"
camera_env_dir = os.path.join(dataset_root, "camera_envs")

info_dir = os.path.join(camera_env_dir, env_flag, "info")
video_dir = os.path.join(camera_env_dir, env_flag, "video")

if not os.path.exists(info_dir):
    os.mkdir(info_dir)

video_list = os.listdir(video_dir)
video_list = sorted(video_list, key=lambda x: int(x.split(".")[0]))
# video_list = list(filter(lambda x: 339 <= int(x.split(".")[0]) < 399, video_list))
# print(video_list)
for video_path_name in tqdm(video_list):
    video_id = video_path_name.split(".")[0]
    video_path = os.path.join(video_dir, video_path_name)
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = int(frame_count / fps)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    raw_info_txt = os.path.join(
        camera_env_dir, "streamline", "info", f"{video_id}.txt")
    scenename = open(raw_info_txt, "r").readlines()[0].split("-")[0]
    info_path = os.path.join(info_dir, f"{video_id}.txt")
    with open(info_path, "w") as f:
        f.write(f"{scenename}-{width}-{height}-{fps}-{frame_count}-{duration}")
