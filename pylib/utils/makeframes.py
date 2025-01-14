import os 
import cv2
import random
import argparse
from tqdm import tqdm

parse = argparse.ArgumentParser()
parse.add_argument("--data_root", type=str, default="./")
parse.add_argument("--data_name", type=str, default="amsterdam")
parse.add_argument("--data_type", type=str, default="train")
args = parse.parse_args()

dataRoot = f"{args.data_root}/{args.data_name}/{args.data_type}"
videoDir = os.path.join(dataRoot, "video")
frameDir = os.path.join(dataRoot, "frames")

if not os.path.exists(frameDir):
    os.makedirs(frameDir)
    
videoPathNameList = os.listdir(videoDir)
videoPathNameList = random.sample(videoPathNameList, 500)
for videoPathName in tqdm(videoPathNameList):
    videoID = videoPathName.split(".")[0]
    videoPath = os.path.join(videoDir, videoPathName)
    framePath = os.path.join(frameDir, videoID)
    if not os.path.exists(framePath):
        os.makedirs(framePath)
    videoCapture = cv2.VideoCapture(videoPath)
    success, frame = videoCapture.read()
    frameIdx = 0
    while success:
        framePathName = os.path.join(framePath, f"{frameIdx:06d}.jpg")
        cv2.imwrite(framePathName, frame)
        success, frame = videoCapture.read()
        frameIdx += 1        
    videoCapture.release()