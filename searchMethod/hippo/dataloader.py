import os
import cv2
import torch
import random
import numpy as np
import pandas as pd
import torchvision.transforms as T
from transformers import VideoMAEImageProcessor
from torch.utils.data import Dataset, DataLoader
from utils import get_state, get_reward, generate_action_from_config


def generator_memory_pool(memory_pool, batch_size):
    sts, rts, cts, stp1s = [], [], [], []
    for _ in range(batch_size):
        track_id = random.choice(list(range(len(memory_pool))))
        step_id = random.choice(
            list(range(1, len(memory_pool[track_id])-1)))  # (st, rt, at, st+1)
        st = get_state(memory_pool[track_id][step_id]["States"])
        stp1 = get_state(memory_pool[track_id][step_id+1]["States"])
        rt = get_reward(memory_pool[track_id][0]["Latency"], memory_pool[track_id][0]["Accuracy"],
                        memory_pool[track_id][step_id]["Latency"], memory_pool[track_id][step_id]["Accuracy"],
                        memory_pool[track_id][step_id-1]["Latency"], memory_pool[track_id][step_id-1]["Accuracy"])
        ct = generate_action_from_config(
            memory_pool[track_id][step_id]["Config"])
        sts.append(st)
        rts.append(rt)
        cts.append(ct)
        stp1s.append(stp1)
    sts = torch.FloatTensor(sts)
    rts = torch.FloatTensor(rts)
    cts = torch.FloatTensor(cts)
    stp1s = torch.FloatTensor(stp1s)
    return sts, rts, cts, stp1s


def generator(replay_memory, batch_size):
    batch, idx = replay_memory.sample(batch_size)

    states, next_states, actions, rewards, terminates = [], [], [], [], []
    for batch_data in batch:
        states.append(batch_data[0])
        next_states.append(batch_data[3])
        actions.append(batch_data[1].tolist())
        rewards.append(batch_data[2])
        terminates.append(batch_data[4])

    states = np.array(states)
    next_states = np.array(next_states)
    actions = np.array(actions)
    rewards = np.array(rewards)
    terminates = np.array(terminates)

    return idx, states, next_states, actions, rewards, terminates


class VideoDataset(Dataset):
    def __init__(self, video_paths, track_datas,
                 framenumber, framegap, objects=['car', 'bus', 'truck'], transform=None, flag='train'):
        self.video_paths = video_paths
        self.track_datas = track_datas
        self.objects = objects
        self.transform = transform
        self.framenumber = framenumber
        self.framegap = framegap
        self.flag = flag

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        track_data = self.track_datas[idx]
        cap = cv2.VideoCapture(video_path)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Make sure the video has at least self.framegap * self.framenumber frames
        if total_frames < self.framegap * self.framenumber:
            return None

        # Select a random start index
        if self.flag == 'train':
            start_frame = random.randint(
                0, total_frames - self.framegap * self.framenumber)
        elif self.flag == 'test':
            start_frame = 0

        frames = []
        sel, agg, count = {object_name: [] for object_name in self.objects}, \
            {object_name: [] for object_name in self.objects}, \
            {object_name: 0 for object_name in self.objects}
        for frame_id in range(start_frame, start_frame + self.framegap * self.framenumber, self.framegap):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if self.transform:
                    frame = self.transform(frame)
                frames.append(frame)
                framedata = track_data[frame_id]
                for object_name in self.objects:
                    if framedata is None:
                        sel[object_name].append(0)
                        agg[object_name].append(0)
                        continue
                    objectdata = [
                        fd for fd in framedata if fd['class'] == object_name]
                    sel[object_name].append(int(len(objectdata) != 0))
                    agg[object_name].append(len(objectdata))
                    count[object_name] += len(objectdata)
            else:
                break

        cap.release()

        frames = torch.stack(frames)

        return frames, sel, agg, count, idx


class VideoDataset2(Dataset):
    def __init__(self, video_paths, track_datas,
                 framenumber, framegap, objects=['car', 'bus', 'truck'], transform=None, flag='train',
                 data_dir="/mnt/data_ssd1/lzp/otif-dataset/dataset/hippo", context_path_name="context_norm.csv"):
        self.video_paths = video_paths
        self.track_datas = track_datas
        self.objects = objects
        self.transform = transform
        self.framenumber = framenumber
        self.framegap = framegap
        self.flag = flag
        self.video_ids = [int(video_path.split('/')[-1].split('.')[0])
                          for video_path in video_paths]
        if data_dir is not None:
            context_path = os.path.join(
                data_dir, f"{flag}/{context_path_name}")
            print(f"\033[35mLoading context from {context_path} ... ...\033[0m")
            self.contexts = self.load_context(context_path)
        else:
            self.contexts = None

    def __len__(self):
        return len(self.video_paths)

    def load_context(self, context_path):
        context = pd.read_csv(context_path)
        contexts = {}
        for _, row in context.iterrows():
            contexts[row['Video']] = row
        return contexts

    def getitem_context(self, video_id):
        return self.contexts[video_id]

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        track_data = self.track_datas[idx]
        cap = cv2.VideoCapture(video_path)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Make sure the video has at least self.framegap * self.framenumber frames
        if total_frames < self.framegap * self.framenumber:
            return None

        # Select a random start index
        if self.flag == 'train':
            start_frame = random.randint(
                0, total_frames - self.framegap * self.framenumber)
        elif self.flag == 'test':
            start_frame = 0

        frames = []
        sel, agg, count = {object_name: [] for object_name in self.objects}, \
            {object_name: [] for object_name in self.objects}, \
            {object_name: 0 for object_name in self.objects}
        for frame_id in range(start_frame, start_frame + self.framegap * self.framenumber, self.framegap):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if self.transform:
                    frame = self.transform(frame)
                frames.append(frame)
                framedata = track_data[frame_id]
                for object_name in self.objects:
                    if framedata is None:
                        sel[object_name].append(0)
                        agg[object_name].append(0)
                        continue
                    objectdata = [
                        fd for fd in framedata if fd['class'] == object_name]
                    sel[object_name].append(int(len(objectdata) != 0))
                    agg[object_name].append(len(objectdata))
                    count[object_name] += len(objectdata)
            else:
                break

        cap.release()

        frames = torch.stack(frames)
        return frames


def build_transform(model_ckpt="/mnt/data_ssd1/lzp/MCG_NJUvideomae_base"):
    image_processor = VideoMAEImageProcessor.from_pretrained(model_ckpt)
    mean = image_processor.image_mean
    std = image_processor.image_std

    if "shortest_edge" in image_processor.size:
        height = width = image_processor.size["shortest_edge"]
    else:
        height = image_processor.size["height"]
        width = image_processor.size["width"]
    resize_to = (height, width)

    transform = T.Compose([T.ToPILImage(),
                           T.Resize(resize_to),
                           T.ToTensor(),
                           T.Normalize(mean,
                                       std)])
    return transform
