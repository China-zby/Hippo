import os
import pickle
from collections import Counter

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

from utils import read_json


class Trajectory(object):
    def __init__(self, data):
        self.data = data
        self.info = self.analysis_data()

    def analysis_data(self):
        min_area, max_area = None, None
        min_speed, max_speed = None, None
        min_acc, max_acc = None, None
        min_turn, max_turn = None, None
        min_turn_speed, max_turn_speed = None, None
        last_speed = None

        start_point, end_point = self.data[0], self.data[-1]
        start_xcenter, start_ycenter = (
            start_point[0] + start_point[2]) / 2, (start_point[1] + start_point[3]) / 2
        end_xcenter, end_ycenter = (
            end_point[0] + end_point[2]) / 2, (end_point[1] + end_point[3]) / 2
        dist = ((start_xcenter - end_xcenter) ** 2 +
                (start_ycenter - end_ycenter) ** 2) ** 0.5
        turn = [end_xcenter - start_xcenter, end_ycenter - start_ycenter]
        movetrack = True
        if dist < 10:
            movetrack = False

        for i in range(len(self.data) - 2):
            left1, top1, right1, bottom1 = self.data[i]
            xcenter1, ycenter1 = (left1 + right1) / 2, (top1 + bottom1) / 2
            left2, top2, right2, bottom2 = self.data[i + 1]
            xcenter2, ycenter2 = (left2 + right2) / 2, (top2 + bottom2) / 2
            area1 = (right1 - left1) * (bottom1 - top1)
            area2 = (right2 - left2) * (bottom2 - top2)
            if min_area is None or area2 < min_area:
                min_area = area2
            if max_area is None or area2 > max_area:
                max_area = area2

            if movetrack:
                speed = ((xcenter2 - xcenter1) ** 2 +
                         (ycenter2 - ycenter1) ** 2) ** 0.5
                if min_speed is None or speed < min_speed:
                    min_speed = speed
                if max_speed is None or speed > max_speed:
                    max_speed = speed

                if last_speed is not None:
                    last_speed_x, last_speed_y = last_speed
                    speed_x, speed_y = xcenter2 - xcenter1, ycenter2 - ycenter1
                    acc = ((speed_x - last_speed_x) ** 2 +
                           (speed_y - last_speed_y) ** 2) ** 0.5
                    if speed_x ** 2 + speed_y ** 2 == 0:
                        turn = 0
                        turn_speed = 0
                    else:
                        turn = abs(speed_x * last_speed_y - speed_y *
                                   last_speed_x) / (speed_x ** 2 + speed_y ** 2)
                        turn_speed = turn / speed
                    if min_acc is None or acc < min_acc:
                        min_acc = acc
                    if max_acc is None or acc > max_acc:
                        max_acc = acc
                    if min_turn is None or turn < min_turn:
                        min_turn = turn
                    if max_turn is None or turn > max_turn:
                        max_turn = turn
                    if min_turn_speed is None or turn_speed < min_turn_speed:
                        min_turn_speed = turn_speed
                    if max_turn_speed is None or turn_speed > max_turn_speed:
                        max_turn_speed = turn_speed

                last_speed = [xcenter2 - xcenter1, ycenter2 - ycenter1]

        return {"min_area": min_area if min_area is not None else 0,
                "max_area": max_area if max_area is not None else 0,
                "min_speed": min_speed if min_speed is not None else 0,
                "max_speed": max_speed if max_speed is not None else 0,
                "min_acc": min_acc if min_acc is not None else 0,
                "max_acc": max_acc if max_acc is not None else 0,
                "min_turn": min_turn if min_turn is not None else 0,
                "max_turn": max_turn if max_turn is not None else 0,
                "min_turn_speed": min_turn_speed if min_turn_speed is not None else 0,
                "max_turn_speed": max_turn_speed if max_turn_speed is not None else 0,
                "dist": dist, "turn": turn}

class CameraCluster(object):
    def __init__(self, cameraList=None, n_clusters=2,
                 mean=None, std=None):
        self.cameraList = cameraList
        self.n_clusters = n_clusters
        self.labels_ = None  # To store cluster labels for each camera
        self.mean = mean
        self.std = std
        if cameraList is not None:
            self._cluster()

    def _cluster(self):
        # Extracting feature vectors from cameraList
        feature_vectors = np.asarray(
            [camera.representation for camera in self.cameraList])

        features = (feature_vectors - self.mean) / self.std

        # Creating GMM object
        gmm = GaussianMixture(n_components=self.n_clusters, random_state=0)

        # Fitting the model
        gmm.fit(features)

        # Getting the cluster assignments
        self.labels_ = gmm.predict(features)

        counter = Counter(self.labels_)

        infos = []
        for item, count in counter.items():
            infos.append(f"Element: {item}, Count: {count}\n")
        self.info = "".join(infos)

        camera_cluster, camera_map = {}, {}
        for camera, label in enumerate(self.labels_):
            if label not in camera_cluster:
                camera_cluster[label] = []
            camera_cluster[label].append(camera)
            camera_map[camera] = label
        self.camera_map = camera_map
        self.camera_cluster = camera_cluster

    def __call__(self, camera):
        return self.camera_map[camera]

    def __repr__(self):
        return self.info

    def save(self, path):
        pickle.dump([self.cameraList,
                     self.n_clusters,
                     self.labels_,
                     self.info,
                     self.camera_map,
                     self.camera_cluster], open(path, "wb"))

    def load(self, path):
        self.cameraList, self.n_clusters, self.labels_, self.info, self.camera_map, self.camera_cluster = pickle.load(
            open(path, "rb"))


class LSTMExtractor(nn.Module):
    def __init__(self):
        self.lstm = nn.LSTM(4, 64, bidirectional=True, batch_first=True)

    def learn_track_representations(self, tracks):
        pass


class TDataset(torch.utils.data.Dataset):
    def __init__(self, trackdir):
        self.trackdir = trackdir
        self.dataset = self.make_dataset(trackdir)

    def make_dataset(self, trackdir):
        dataset = []
        trackmapdata = {}
        trackpathnames = os.listdir(trackdir)
        for trackpathname in tqdm(trackpathnames[:10]):
            trackpath = os.path.join(trackdir, trackpathname)
            trackdata = read_json(trackpath)
            if trackdata is None or len(trackdata) == 0:
                continue
            for trackline in trackdata:
                if trackline is None or len(trackline) == 0:
                    continue
                for track in trackline:
                    trackid = track["track_id"]
                    if f"{trackpathname}_{trackid}" not in trackmapdata:
                        trackmapdata[f"{trackpathname}_{trackid}"] = []
                    trackmapdata[f"{trackpathname}_{trackid}"].append([track['left'],  track['top'],
                                                                       track['right'], track['bottom']])
        trackid = 0
        for nameid in trackmapdata:
            if len(trackmapdata[nameid]) < 32:
                continue
            dataset.append(trackmapdata[nameid])
            trackid += 1
        return dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        track = self.dataset[idx]
