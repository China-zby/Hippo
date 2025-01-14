import os
import cv2
import json
import numpy as np


def visual_track_data(track_data, prefix, suffix,
                      prefix_frame_gap, suffix_frame_gap,
                      width, height, total_frame, tid, scene_name):
    blank_image = np.ones((height, width, 3), dtype=np.uint8) * 255
    for point in track_data:
        left, top, right, bottom, frame_id = point
        left, top, right, bottom = int(left), int(top), int(right), int(bottom)
        cv2.rectangle(blank_image, (left, top),
                      (right, bottom), (0, 0, 255), 1)
        cv2.putText(blank_image, str(frame_id), (left, top),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    left, top, right, bottom = prefix
    left, top, right, bottom = int(left), int(top), int(right), int(bottom)
    cv2.rectangle(blank_image, (left, top), (right, bottom), (0, 255, 0), 1)
    cv2.putText(blank_image, "prefix-" + str(prefix_frame_gap),
                (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    left, top, right, bottom = suffix
    left, top, right, bottom = int(left), int(top), int(right), int(bottom)
    cv2.rectangle(blank_image, (left, top), (right, bottom), (255, 0, 0), 1)
    cv2.putText(blank_image, "suffix-" + str(suffix_frame_gap),
                (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    if not os.path.exists(f"./prefix_suffix_outputs/{scene_name}"):
        os.mkdir(f"./prefix_suffix_outputs/{scene_name}")
    cv2.imwrite(f"./prefix_suffix_outputs/{scene_name}/{tid}.jpg", blank_image)


def visual_gtandpred_track_data(track_data,
                                pred_prefix, pred_suffix,
                                prefix, suffix,
                                width, height, total_frame, tid, scene_name):
    # print gt track
    blank_image = np.ones((height, width, 3), dtype=np.uint8) * 255
    for point in track_data:
        left, top, right, bottom, frame_id = point
        left, top, right, bottom = int(left), int(top), int(right), int(bottom)
        cv2.rectangle(blank_image, (left, top),
                      (right, bottom), (0, 0, 255), 1)
        cv2.putText(blank_image, str(frame_id), (left, top),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    left, top, right, bottom, start_gap = prefix
    left, top, right, bottom, start_gap = int(left), int(
        top), int(right), int(bottom), int(start_gap)
    cv2.rectangle(blank_image, (left, top), (right, bottom), (0, 255, 0), 1)
    cv2.putText(blank_image, "prefix-" + str(start_gap),
                (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    left, top, right, bottom, end_gap = suffix
    left, top, right, bottom, end_gap = int(left), int(
        top), int(right), int(bottom), int(end_gap)
    cv2.rectangle(blank_image, (left, top), (right, bottom), (255, 0, 0), 1)
    cv2.putText(blank_image, "suffix-" + str(end_gap),
                (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    if not os.path.exists(f"./gtandpred_outputs/{scene_name}"):
        os.mkdir(f"./gtandpred_outputs/{scene_name}")
    cv2.imwrite(f"./gtandpred_outputs/{scene_name}/{tid}.jpg", blank_image)

    # print pred track
    blank_image = np.ones((height, width, 3), dtype=np.uint8) * 255
    for point in track_data:
        left, top, right, bottom, frame_id = point
        left, top, right, bottom = int(left), int(top), int(right), int(bottom)
        cv2.rectangle(blank_image, (left, top),
                      (right, bottom), (0, 0, 255), 1)
        cv2.putText(blank_image, str(frame_id), (left, top),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    left, top, right, bottom, start_gap = pred_prefix
    left, top, right, bottom, start_gap = int(left), int(
        top), int(right), int(bottom), int(start_gap)
    cv2.rectangle(blank_image, (left, top), (right, bottom), (0, 255, 0), 1)
    cv2.putText(blank_image, "prefix-" + str(start_gap),
                (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    left, top, right, bottom, end_gap = pred_suffix
    left, top, right, bottom, end_gap = int(left), int(
        top), int(right), int(bottom), int(end_gap)
    cv2.rectangle(blank_image, (left, top), (right, bottom), (255, 0, 0), 1)
    cv2.putText(blank_image, "suffix-" + str(end_gap),
                (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    if not os.path.exists(f"./gtandpred_outputs/{scene_name}"):
        os.mkdir(f"./gtandpred_outputs/{scene_name}")
    cv2.imwrite(
        f"./gtandpred_outputs/{scene_name}/{tid}_pred.jpg", blank_image)
