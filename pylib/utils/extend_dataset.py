import shutil
import os

def copy_and_rename_files(src_folder, dst_folder, extension, start_index, end_index):
    """
    复制并重命名文件
    """
    for idx in range(start_index, end_index + 1):
        src_file = os.path.join(src_folder, f"{idx % 399}.{extension}")
        dst_file = os.path.join(dst_folder, f"{idx}.{extension}")
        shutil.copy(src_file, dst_file)


def main():
    video_src_folder = "/home/lzp/otif-dataset/dataset/camera_envs/streamline/video"
    track_src_folder = "/home/lzp/otif-dataset/dataset/camera_envs/streamline/tracks"
    
    if not os.path.exists(video_src_folder):
        os.makedirs(video_src_folder)
    
    if not os.path.exists(track_src_folder):
        os.makedirs(track_src_folder)

    # 视频文件
    copy_and_rename_files(video_src_folder, video_src_folder, 'mp4', 399, 999)

    # 标注文件
    for extension in ['txt', 'json', 'pkl']:
        copy_and_rename_files(track_src_folder, track_src_folder, extension, 399, 999)


if __name__ == "__main__":
    main()
