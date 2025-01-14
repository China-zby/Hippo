import os

import cv2
import argparse
import numpy as np
from tqdm import tqdm

parse = argparse.ArgumentParser()
parse.add_argument("--testmode", type=str, default="baseline")
parse.add_argument("--gtpath", type=str, default="./")
parse.add_argument("--trackerpath", type=str, default="./")
parse.add_argument("--dataset_name", type=str, default="amsterdam")
parse.add_argument("--skipframe", type=int, default=1)
parse.add_argument("--dataroot", type=str, default="./")
args = parse.parse_args()

video_ids = list(map(lambda x: int(
    x.rsplit('.')[0].split('-')[1]), os.listdir(args.trackerpath)))
print(f"Total video of {args.dataroot}: ", len(video_ids))
for video_id in tqdm(video_ids):
    rescale = 1
    frameTrackerDict = {}
    tracker_path = os.path.join(
        args.trackerpath, f"{args.dataset_name}S{args.skipframe}-{video_id}.txt")
    with open(tracker_path, "r") as freader:
        content = freader.read()
        lines = content.split("\n")
        for line in lines:
            if line is not None and len(line) > 0:
                line = line.split(",")
                frameid, trackid, x1, y1, width, height, score, classid = int(line[0]), int(line[1]), int(
                    line[2]), int(line[3]), int(line[4]), int(line[5]), float(line[6]), int(line[7])
                if frameid not in frameTrackerDict:
                    frameTrackerDict[frameid] = []
                frameTrackerDict[frameid].append(
                    [trackid, x1, y1, width, height, score, classid])

    GTTrackerDict = {}
    gt_path = os.path.join(
        args.gtpath, f"{args.dataset_name}S{args.skipframe}-{video_id}/gt/gt.txt")
    with open(gt_path, "r") as freader:
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
                    [trackid, x1, y1, width, height, score, classid])

    videopath = os.path.join(
        args.dataroot, f"{args.testmode}/video/{video_id}.mp4")
    videoCapture = cv2.VideoCapture(videopath)
    labeled_video_dir = os.path.join(
        os.path.dirname(args.trackerpath), "video_data")
    if not os.path.exists(labeled_video_dir):
        os.makedirs(labeled_video_dir)
    outputVideoPath = os.path.join(os.path.dirname(
        args.trackerpath), f"video_data/{args.dataset_name}S{args.skipframe}-{video_id}.mp4")
    frameRate = max(videoCapture.get(cv2.CAP_PROP_FPS) // args.skipframe, 1)
    frameWidth = videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)
    frameHeight = videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frameNumber = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)

    videoWriter = cv2.VideoWriter(outputVideoPath,
                                  cv2.VideoWriter_fourcc(*'mp4v'),
                                  frameRate, (int(frameWidth//rescale) * 2, int(frameHeight//rescale)))
    frameid = 0
    color_dict = {2: (178, 167, 110),
                  5: (71, 171, 167),
                  7: (170, 150, 255)}
    while True:
        ret, frame = videoCapture.read()
        if not ret:
            break
        if frameid % args.skipframe != 0:
            frameid += 1
            continue
        remapframeid = int(frameid / args.skipframe) + 1
        if remapframeid in frameTrackerDict:
            tracker_frame = frame.copy()
            for trackid, x1, y1, width, height, score, classid in frameTrackerDict[remapframeid]:
                cv2.rectangle(tracker_frame, (x1, y1), (x1+width,
                              y1+height), color_dict[classid], 2)
                cv2.putText(tracker_frame, f"{trackid}-{classid}", (x1, y1),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_dict[classid], 2)
        else:
            tracker_frame = frame.copy()
        cv2.putText(tracker_frame, "Tracked Result", (0, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
        if remapframeid in GTTrackerDict:
            gt_frame = frame.copy()
            for trackid, x1, y1, width, height, score, classid in GTTrackerDict[remapframeid]:
                cv2.rectangle(gt_frame, (x1, y1), (x1+width,
                              y1+height), color_dict[classid], 2)
                cv2.putText(gt_frame, f"{trackid}-{classid}", (x1, y1),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_dict[classid], 2)
        else:
            gt_frame = frame.copy()
        cv2.putText(gt_frame, "Ground Truth", (0, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
        tracker_frame = cv2.resize(
            tracker_frame, (int(frameWidth//rescale), int(frameHeight//rescale)))
        gt_frame = cv2.resize(
            gt_frame, (int(frameWidth//rescale), int(frameHeight//rescale)))
        frame = np.concatenate((gt_frame, tracker_frame), axis=1)
        videoWriter.write(frame)
        frameid += 1
    videoCapture.release()
    videoWriter.release()
