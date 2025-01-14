import os
import shutil
import random

# 定义源文件夹和目标文件夹
src_dir = '/home/lzp/otif-dataset/dataset/camera_envs/streamline'
dst_dir = '/home/lzp/otif-dataset/dataset/camera_envs/real'

# 如果目标文件夹不存在，创建它
if not os.path.exists(dst_dir):
    os.mkdir(dst_dir)

# 定义子文件夹
sub_dirs = ['video', 'info', 'tracks']

# 在目标文件夹中创建子文件夹
for sub_dir in sub_dirs:
    if not os.path.exists(os.path.join(dst_dir, sub_dir)):
        os.mkdir(os.path.join(dst_dir, sub_dir))

# 获取所有视频文件
video_files = [f for f in os.listdir(
    os.path.join(src_dir, 'video')) if f.endswith('.mp4')]

# 打乱视频文件列表
random.shuffle(video_files)

# 复制并重命名文件
for idx, video_file in enumerate(video_files):
    video_src_path = os.path.join(src_dir, 'video', video_file)
    video_dst_path = os.path.join(dst_dir, 'video', f"{idx}.mp4")
    shutil.copy(video_src_path, video_dst_path)

    # 复制info文件
    info_src_path = os.path.join(
        src_dir, 'info', video_file.replace('.mp4', '.txt'))
    info_dst_path = os.path.join(dst_dir, 'info', f"{idx}.txt")
    if os.path.exists(info_src_path):
        shutil.copy(info_src_path, info_dst_path)

    # 复制tracks文件
    for ext in ['.txt', '.json', '.pkl']:
        track_src_path = os.path.join(
            src_dir, 'tracks', video_file.replace('.mp4', ext))
        track_dst_path = os.path.join(dst_dir, 'tracks', f"{idx}{ext}")
        shutil.copy(track_src_path, track_dst_path)

print("复制和重命名完成!")
