import os
import cv2
import json
import torch
import random
from tqdm import tqdm

from utils import visual_track_data


class TrackDataset():
    def __init__(self, data_dir, track_path_list, scene2id):
        self.data_path = data_dir
        self.scene2id = scene2id
        self.frame_norm = 1800
        self.id2scene = {v: k for k, v in scene2id.items()}
        self.dataset = self.load_dataset(data_dir, track_path_list)

    def load_dataset(self, data_dir, track_path_list):
        dataset = []
        max_lenth = 0
        for track_path in tqdm(track_path_list):
            assert track_path.endswith(".json")
            "Track file must be json file"
            track_path = os.path.join(data_dir, track_path)
            video_info_path = track_path.replace(
                "tracks", "info").replace(".json", ".txt")

            with open(video_info_path, "r") as reader:
                video_info = reader.readlines()[0]
                scene_name, width, height = video_info.split("-")
                width, height = int(width), int(height)
            frame_data = json.load(open(track_path, 'r'))

            total_frame = len(frame_data)

            track_data = {}
            for frame_id in range(len(frame_data)):
                frame_info = frame_data[frame_id]
                if frame_info is None or len(frame_info) == 0:
                    continue
                for track_info in frame_info:  # left, top, right, bottom, track_id, class, score
                    track_id = track_info["track_id"]
                    if track_id not in track_data:
                        track_data[track_id] = []
                    track_info["frame_id"] = frame_id
                    track_data[track_id].append(track_info)

            # 1. filter static track
            track_data = self.filter_static_track(track_data)
            # 2. filter short track
            track_data = self.filter_short_track(track_data)
            # 3. filter track that don't have the first frame or the last frame
            track_data = self.filter_track_without_first_or_last(
                track_data, width, height)

            for track_id in list(track_data.keys()):
                max_lenth = max(max_lenth, len(track_data[track_id]))
                dataset.append(
                    [track_data[track_id], self.scene2id[scene_name], width, height, total_frame])
        return dataset

    def filter_static_track(self, track_data):
        new_track_data = {}
        abs_distance = 200
        for track_id in track_data.keys():
            track_info = track_data[track_id]
            if len(track_info) < 2:
                continue
            start_point = [(track_info[0]["left"] + track_info[0]["right"])/2.0,
                           (track_info[0]["top"] + track_info[0]["bottom"])/2.0]
            end_point = [(track_info[-1]["left"] + track_info[-1]["right"])/2.0,
                         (track_info[-1]["top"] + track_info[-1]["bottom"])/2.0]
            if abs(start_point[0] - end_point[0]) > abs_distance or \
                    abs(start_point[1] - end_point[1]) > abs_distance:
                new_track_data[track_id] = track_info
        return new_track_data

    def filter_short_track(self, track_data):
        new_track_data = {}
        min_track_length = 128
        for track_id in track_data.keys():
            track_info = track_data[track_id]
            if len(track_info) >= min_track_length:
                new_track_data[track_id] = track_info
        return new_track_data

    def filter_track_without_first_or_last(self, track_data, width, height):
        rates = 0
        new_track_data = {}
        for track_id in track_data.keys():
            start_point, end_point = track_data[track_id][0], track_data[track_id][-1]
            if self.if_in_bound(width, height, start_point) and self.if_in_bound(width, height, end_point):
                new_track_data[track_id] = track_data[track_id]
                rates += 1
        return new_track_data

    def if_in_bound(self, width, height, bbox):
        # bbox : {left, top, right, bottom}
        abs_distance = 200
        x1, y1, x2, y2 = bbox["left"], bbox["top"], bbox["right"], bbox["bottom"]
        if abs(x1) < abs_distance or abs(y1) < abs_distance or abs(x2 - width) < abs_distance or abs(y2 - height) < abs_distance or \
                x1 < 0 or y1 < 0 or x2 > width or y2 > height:
            return True
        else:
            return False

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        max_length = 1800
        sampling_rates = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        track_data, scene_id, width, height, total_frame = self.dataset[idx]
        # 1. get the prefix and suffix point
        prefix_point, suffix_point = track_data[0], track_data[-1]
        sampling_rate = random.choice(
            [sampling_rate for sampling_rate in sampling_rates if len(track_data) > sampling_rate * 3])
        track_data = track_data[::sampling_rate]

        min_segment_length = 1
        max_segment_length = len(track_data) - 2

        segment_length = random.randint(min_segment_length, max_segment_length)

        start_index = random.randint(1, len(track_data) - segment_length - 1)

        track_data = track_data[start_index:start_index + segment_length]

        # 2. get the prefix and suffix frame gap
        prefix_frame_gap,  suffix_frame_gap = track_data[0]["frame_id"] - \
            prefix_point["frame_id"], suffix_point["frame_id"] - \
            track_data[-1]["frame_id"]
        prefix_position, suffix_position = [prefix_point["left"], prefix_point["top"],
                                            prefix_point["right"], prefix_point["bottom"]], [suffix_point["left"], suffix_point["top"],
                                                                                             suffix_point["right"], suffix_point["bottom"]]
        # 3. get the trajectory length
        trajectory_length = len(track_data)
        # 4. get the trajectory
        trajectory = []
        for point in track_data:
            relative_frame_id = point["frame_id"] - track_data[0]["frame_id"]
            trajectory.append([point["left"] / width, point["top"] / height,
                               point["right"] /
                               width, point["bottom"] / height,
                               relative_frame_id / self.frame_norm])
        # visual_track_data(trajectory,
        #                   prefix_position, suffix_position,
        #                   prefix_frame_gap, suffix_frame_gap,
        #                   width, height, total_frame, idx, self.id2scene[scene_id])
        # 5. padding
        if trajectory_length < max_length:
            for i in range(max_length - trajectory_length):
                trajectory.append([0, 0, 0, 0, 0])
        else:
            trajectory = trajectory[:max_length]
        trajectory = torch.FloatTensor(trajectory)
        # 6. get the prefix and suffix
        prefix = torch.FloatTensor([prefix_position[0] / width, prefix_position[1] / height,
                                    prefix_position[2] /
                                    width, prefix_position[3] / height,
                                    prefix_frame_gap / self.frame_norm])
        suffix = torch.FloatTensor([suffix_position[0] / width, suffix_position[1] / height,
                                    suffix_position[2] /
                                    width, suffix_position[3] / height,
                                    suffix_frame_gap / self.frame_norm])
        # 7. get the scene id
        scene_id = torch.tensor([scene_id])
        return trajectory, prefix, suffix, scene_id, trajectory_length
