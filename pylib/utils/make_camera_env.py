import os
import cv2
import subprocess

# camera_envs = ['']
env_flag = "train"
suffixs = ["json"]
camera_flags = ['train', 'valid', 'tracker']
dataset_root = "/mnt/data_ssd1/lzp/otif-dataset/dataset"
camera_env_dir = os.path.join(dataset_root, "camera_envs")
camera_scenes = os.listdir(dataset_root)
camera_scenes = [scene for scene in camera_scenes if not "env" in scene]

if not os.path.exists(camera_env_dir):
    os.mkdir(camera_env_dir)

camera_streamline_dir = os.path.join(camera_env_dir, env_flag)
if not os.path.exists(camera_streamline_dir):
    os.mkdir(camera_streamline_dir)
if not os.path.exists(os.path.join(camera_streamline_dir, "video")):
    os.mkdir(os.path.join(camera_streamline_dir, "video"))
if not os.path.exists(os.path.join(camera_streamline_dir, "tracks")):
    os.mkdir(os.path.join(camera_streamline_dir, "tracks"))
if not os.path.exists(os.path.join(camera_streamline_dir, "info")):
    os.mkdir(os.path.join(camera_streamline_dir, "info"))

env_videoid_map = {}
for di, data_flag in enumerate(camera_flags):
    for ci, camera_scene in enumerate(camera_scenes):
        camera_videos = os.listdir(f"{dataset_root}/{camera_scene}/{data_flag}/video")
        for vi, video_file_name in enumerate(camera_videos):
            video_id = int(video_file_name.split(".")[0])
            camera_dir = f"{dataset_root}/{camera_scene}/{data_flag}"
            copy_flags = [os.path.exists(f"{camera_dir}/video/{video_id}.mp4")] +\
                         [os.path.exists(f"{camera_dir}/tracks/{video_id}.{suffix}") for suffix in suffixs]
            if all(copy_flags):
                if f"{ci}-{di}-{vi}" not in env_videoid_map:
                    env_videoid_map[f"{ci}-{di}-{vi}"] = len(env_videoid_map)
                env_video_id = env_videoid_map[f"{ci}-{di}-{vi}"]
                command = f"cp {camera_dir}/video/{video_id}.mp4 {camera_streamline_dir}/video/{env_video_id}.mp4"
                process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                output, error = process.communicate()
                for suffix in suffixs:
                    command = f"cp {camera_dir}/tracks/{video_id}.{suffix} {camera_streamline_dir}/tracks/{env_video_id}.{suffix}"
                    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    output, error = process.communicate()
                with open(f"{camera_dir}/video/{video_id}.mp4") as f:
                    reader = cv2.VideoCapture(f"{camera_dir}/video/{video_id}.mp4")
                    width, height = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH)), int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
                with open(f"{camera_streamline_dir}/info/{env_video_id}.txt", "w") as f:
                    f.write(f"{camera_scene}-{width}-{height}")

# if not os.path.exists(camera_streamline_dir):
#     os.mkdir(camera_streamline_dir)
# if not os.path.exists(os.path.join(camera_streamline_dir, "seg-train/images")):
#     os.makedirs(os.path.join(camera_streamline_dir, "seg-train/images"))

# env_imageid_map = {}
# for ci, camera_scene in enumerate(camera_scenes):
#     camera_images = [ci for ci in os.listdir(f"{dataset_root}/{camera_scene}/train/seg-train/images") if ci.endswith('.jpg')]
#     for ii, image_file_name in enumerate(camera_images):
#         image_id = int(image_file_name.split(".")[0])
#         camera_dir = f"{dataset_root}/{camera_scene}/train/seg-train/images"
#         copy_flags = [os.path.exists(f"{camera_dir}/{image_id}.jpg"),
#                       os.path.exists(f"{camera_dir}/{image_id}.json")]
#         if all(copy_flags):
#             if f"{ci}-{ii}" not in env_imageid_map:
#                 env_imageid_map[f"{ci}-{ii}"] = len(env_imageid_map)
#             env_image_id = env_imageid_map[f"{ci}-{ii}"]
#             command = f"cp {camera_dir}/{image_id}.jpg {camera_streamline_dir}/seg-train/images/{env_image_id}.jpg"
#             process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#             output, error = process.communicate()
            
#             command = f"cp {camera_dir}/{image_id}.json {camera_streamline_dir}/seg-train/images/{env_image_id}.json"
#             process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#             output, error = process.communicate()