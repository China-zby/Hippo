import os
import cv2
import numpy as np

from ultralytics import YOLO
from ultralytics.utils.checks import check_yaml
from ultralytics.utils import IterableSimpleNamespace, yaml_load

from trackers.byte_tracker import BYTETracker

video_name_dict = {"Adventure_Rentals": "adventure",
                   "caldot1": "caldot1",
                   "caldot2": "caldot2",
                   "Flat_Creek_Inn": "flat",
                   "Jackson_Town": "jackson",
                   "shibuya": "shibuya",
                   "Square_Northeast": "square",
                   "Taipei_Hires": "taipei"}

if __name__ == "__main__":
    model = YOLO('yolov8n.pt')
    tracker_cfg = check_yaml("./trackers/cfgs/bytetrack.yaml")
    cfg = IterableSimpleNamespace(**yaml_load(tracker_cfg))
    video_dir = "/home/xuyanchao/video_data/Collected_live_videos/standard_split"
    mask_image_dir = "/home/xuyanchao/video_data/masks"
    video_path_name_list = os.listdir(video_dir)
    skip_number = 16
    
    for video_path_name in video_path_name_list:
        tracker = BYTETracker(args=cfg, frame_rate=min(30 // skip_number, 2))
        
        video_name = video_name_dict[video_path_name]
        video_path = os.path.join(video_dir, video_path_name, f"concat/test/{video_name}_test.mp4")
        image_mask_path = os.path.join(mask_image_dir, f"{video_name}_mask.jpg")

        image_mask = cv2.imread(image_mask_path)
        black_image = np.zeros_like(image_mask)

        video_reader = cv2.VideoCapture(video_path)
        total_frame_number = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
        print("total_frame_number: ", total_frame_number)
        # video_writer = cv2.VideoWriter(f"./videos/{video_name}_track.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (1920, 1080))
        track_writer = open(f"./results/{video_name}_track.txt", "w")
        frame_id = 0
        while True:
            ret, frame = video_reader.read()
            if not ret:
                break
            # copy mask to frame
            frame[image_mask == 0] = black_image[image_mask == 0]
            frame_id += 1
            if (frame_id - 1) % skip_number == 0:
                results = model(frame, conf=0.1, verbose=False, classes=[2,5,7])[0].boxes.cpu().numpy()
                if len(results) == 0:
                    # video_writer.write(frame)
                    continue
                track_results = tracker.update(results)
                if track_results.shape[0] == 0:
                    # video_writer.write(frame)
                    continue
                for track_result in track_results:
                    x1, y1, x2, y2, track_id, score, cls, idx = track_result
                    w, h = x2 - x1, y2 - y1
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = int(w), int(h)
                    track_id = int(track_id)
                    track_writer.write(f"{frame_id},{track_id},{x1},{y1},{w},{h},{score},{cls},-1,-1,-1\n")
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, str(track_id), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                # video_writer.write(frame)
        video_reader.release()
        # video_writer.release()