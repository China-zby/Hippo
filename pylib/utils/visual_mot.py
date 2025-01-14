import cv2
import numpy as np
from alive_progress import alive_bar

vid = 3
srate = 32
data_type = "train"
method = "test_detective_reinforce_all"

input_bbox_path = f"./TrackEval/data/trackers/videodb/hippoS{srate}-{data_type}/{method}/data/hippoS{srate}-{vid}.txt"
# input_bbox_path = f"/home/lzp/go-work/src/otifpipeline/TrackEval/data/gt/videodb/hippo{srate}-{data_type}/hippo{srate}-{vid}/gt/gt.txt"
# input_bbox_path = f"/home/lzp/otif-dataset/dataset/hippo/streamline/tracks/{vid}.txt"
video_path = f"/home/lzp/otif-dataset/dataset/hippo/{data_type}/video/{vid}.mp4"

cap = cv2.VideoCapture(video_path)

videofps = cap.get(cv2.CAP_PROP_FPS)
videoframecount = cap.get(cv2.CAP_PROP_FRAME_COUNT)
print("videofps: ", videofps)
print("videoframecount: ", videoframecount)

GTTrackerDict = {}
with open(input_bbox_path, "r") as freader:
    content = freader.read()
    lines = content.split("\n")
    for line in lines:
        if line is not None and len(line) > 0:
            line = line.split(",")
            frameid, trackid, x1, y1, width, height, score, classid = int(line[0]), int(line[1]), int(
                line[2]), int(line[3]), int(line[4]), int(line[5]), float(line[6]), int(line[7])
            if frameid not in GTTrackerDict:
                GTTrackerDict[frameid] = []
            GTTrackerDict[frameid].append(
                [trackid, x1, y1, width + x1, height + y1, score, classid])

max_frame = max(GTTrackerDict.keys())
VIDEOwidth, VIDEOheight = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# srate = 1
save_path = f"./save_{data_type}_{vid}_{srate}_video.mp4"
newcap = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(
    *'MP4V'), max(1, 30 // srate), (VIDEOwidth, VIDEOheight))
frameid = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    if frameid % srate != 0:
        frameid += 1
        continue
    remapframeid = int(frameid / srate) + 1
    if remapframeid in GTTrackerDict:
        for trackid, x1, y1, x2, y2, score, classid in GTTrackerDict[remapframeid]:
            cv2.rectangle(frame, (x1, y1), (x2, y2),
                          (0, 255, 0), 2)
            cv2.putText(frame, str(trackid), (x1, y1),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        newcap.write(frame)
    frameid += 1
