import os
import json
import cv2


def extract_first_second_of_video(video_path, output_path):
    os.system(f'ffmpeg -i {video_path} -t 1 {output_path}')


def extract_first_second_of_annotation_txt(txt_path, output_path, fps):
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    lines = [line for line in lines if int(line.split(',')[0]) < fps]
    with open(output_path, 'w') as f:
        f.writelines(lines)


def extract_first_second_of_annotation_json(json_path, output_path, fps):
    with open(json_path, 'r') as f:
        data = json.load(f)
    data = data[:fps]
    with open(output_path, 'w') as f:
        json.dump(data, f)


video_dir = '/home/lzp/otif-dataset/dataset/camera_envs/streamline/video'
tracks_dir = '/home/lzp/otif-dataset/dataset/camera_envs/streamline/tracks'
new_video_dir = '/home/lzp/otif-dataset/dataset/camera_envs/one_second/video'
new_tracks_dir = '/home/lzp/otif-dataset/dataset/camera_envs/one_second/tracks'
os.makedirs(new_video_dir, exist_ok=True)

for video_file in os.listdir(video_dir):
    video_path = os.path.join(video_dir, video_file)
    video_id = os.path.splitext(video_file)[0]

    # 从视频中获取帧率
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()

    extract_first_second_of_video(
        video_path, os.path.join(new_video_dir, video_file))
    extract_first_second_of_annotation_txt(os.path.join(tracks_dir, video_id + '.txt'),
                                           os.path.join(
                                               new_tracks_dir, video_id + '.txt'),
                                           fps)
    extract_first_second_of_annotation_json(os.path.join(tracks_dir, video_id + '.json'),
                                            os.path.join(
                                                new_tracks_dir, video_id + '.json'),
                                            fps)
