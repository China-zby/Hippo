import os 
import cv2
import argparse
from tqdm import tqdm

parse = argparse.ArgumentParser()
parse.add_argument("--datasetName", type=str, default="amsterdam")
args = parse.parse_args()

dataRoot = f"./dataset/dataset/{args.datasetName}/"
seperatedVideoDir = f"{dataRoot}/test_separate_video/video/"
fullVideoDir = f"{dataRoot}/test_full_video/video/0.mp4"

videoIds = sorted(list(map(lambda x: int(x.split(".")[0]), os.listdir(seperatedVideoDir))))

for idx, videoID in enumerate(tqdm(videoIds)):
    if idx == 0:
        demoVideo = cv2.VideoCapture(f"{seperatedVideoDir}/{videoID}.mp4")
        frameRate, frameWidth, frameHeight = int(demoVideo.get(cv2.CAP_PROP_FPS)), int(demoVideo.get(cv2.CAP_PROP_FRAME_WIDTH)), int(demoVideo.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
        videoWriter = cv2.VideoWriter(fullVideoDir, cv2.VideoWriter_fourcc(*'mp4v'), frameRate, (frameWidth, frameHeight))
    else: demoVideo = cv2.VideoCapture(f"{seperatedVideoDir}/{videoID}.mp4")
    while True:
        ret, frame = demoVideo.read()
        if not ret:
            break
        videoWriter.write(frame)
    demoVideo.release()
videoWriter.release()
print("Done")