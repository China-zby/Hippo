import os
import time
import json
import torch
import random
import argparse
import numpy as np
import seaborn as sns
from model import RNNModel
from dataloader import datasample
from alive_progress import alive_bar
from matplotlib import pyplot as plt

colors = sns.color_palette("Reds")
val_colors = sns.color_palette("Blues")
train_loss_color = colors[1]
val_loss_color = val_colors[1]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


parses = argparse.ArgumentParser()
parses.add_argument('--seed', type=int, default=2023,
                    help='the training seed')
parses.add_argument('--data_root', type=str, default='./',
                    help='the root path of the dataset')
parses.add_argument('--data_name', type=str,
                    default='amsterdam', help='the name of the dataset')
parses.add_argument('--data_flag', type=str,
                    default='tracker', help='the flag of the dataset')
parses.add_argument('--norm', type=str, default=1000,
                    help='the normalization factor')
parses.add_argument('--save_root', type=str, default="./weights",
                    help='the root path to save the weights')
parses.add_argument('--cfg_path', type=str,
                    default="./cfg_64.json", help='Number of epochs to train')
parses.add_argument('--epochs', type=int, default=9999,
                    help='Number of epochs to train')
parses.add_argument('--max_wait_epochs', type=int,
                    default=25, help='Number of epochs to train')
parses.add_argument('--learning_rate', type=float,
                    default=0.001, help='Learning rate to train')
parses.add_argument('--temporal_threshold', type=float,
                    default=32, help='the temporal threshold')
args = parses.parse_args()

set_seed(args.seed)

if not os.path.exists(os.path.join(args.save_root, "Trackers", args.data_name)):
    os.makedirs(os.path.join(args.save_root, "Trackers", args.data_name))

root_path = os.path.join(args.data_root, args.data_name)
with open(os.path.join(args.cfg_path), 'r') as f:
    cfg = json.load(f)
    skips = cfg['Freqs']
    fps = cfg['FPS']
    MAX_LENGTH = cfg['Max_length']
    NORM = cfg['Norm']
    NUM_BOXES = cfg['Num_boxes']
    BATCH_SIZE = cfg['Batch_size']

learning_rate = args.learning_rate

videos = []
max_track_number = 0
with alive_bar(len(os.listdir(os.path.join(root_path, f'{args.data_flag}/video/')))) as bar:
    for fname in os.listdir(os.path.join(root_path, f'{args.data_flag}/video/')):
        bar()
        if not fname.endswith('.mp4'):
            continue
        videoid = int(fname.split('.mp4')[0])
        with open(os.path.join(root_path, f'{args.data_flag}/tracks', '{}.json'.format(videoid)), 'r') as f:
            detections = json.load(f)
        # get first frame and idx and last frame of all unique tracks
        tracks = {}
        for frame_idx, dlist in enumerate(detections):
            if not dlist:
                continue
            for i, d in enumerate(dlist):
                track_id = d['track_id']
                if track_id not in tracks:
                    tracks[track_id] = []
                tracks[track_id].append((frame_idx, i))

        filtered_tracks = {}
        for track_id, dlist in tracks.items():
            if len(dlist) < 2:
                continue
            dlist = sorted(dlist, key=lambda x: x[0])
            start_frame_id, _ = dlist[0]
            end_frame_id, _ = dlist[-1]
            if end_frame_id - start_frame_id < args.temporal_threshold:
                continue
            filtered_tracks[track_id] = dlist

        tracks = [(track_id, dlist)
                  for track_id, dlist in filtered_tracks.items()]
        if len(tracks) == 0:
            continue
        videos.append((detections, tracks, videoid))
        max_track_number = max(max_track_number, len(tracks))
print("max_track_number: ", max_track_number)
assert max_track_number <= NUM_BOXES
f'NUM_BOXES should be larger than max_track_number'

random.shuffle(videos)
num_val = len(videos)//10
val_videos = videos[0:num_val]
train_videos = videos[num_val:]

model = RNNModel().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=25, gamma=0.8)
model.train()
val_examples = [datasample(val_videos,
                           skips,
                           fps,
                           info_dir=os.path.join(
                               root_path, f'{args.data_flag}/info/'),
                           NORM=NORM,
                           MAX_LENGTH=MAX_LENGTH,
                           NUM_BOXES=NUM_BOXES) for _ in range(32768)]

best_loss = 9999
bad_loss_streak = 0

loss_track, val_loss_track = [], []

for epoch in range(9999):
    start_time = time.time()
    train_losses = []
    model.train()
    with alive_bar(512) as trainbar:
        for _ in range(512):
            examples = [datasample(train_videos,
                                   skips,
                                   fps,
                                   info_dir=os.path.join(
                                       root_path, f'{args.data_flag}/info/'),
                                   NORM=NORM,
                                   MAX_LENGTH=MAX_LENGTH,
                                   NUM_BOXES=NUM_BOXES) for _ in range(BATCH_SIZE)]
            inputs = torch.FloatTensor(np.asarray(
                [example[0] for example in examples])).cuda()
            boxes = torch.FloatTensor(np.asarray(
                [example[1] for example in examples])).cuda()
            mask = torch.FloatTensor(np.asarray(
                [example[2] for example in examples])).cuda()
            targets = torch.FloatTensor(np.asarray(
                [example[3] for example in examples])).cuda()
            loss, outputs = model(inputs,
                                  boxes,
                                  mask,
                                  targets,
                                  MAX_LENGTH=MAX_LENGTH,
                                  BATCH_SIZE=BATCH_SIZE,
                                  NUM_BOXES=NUM_BOXES)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            loss_track.append(min(loss.item(), 2.0))
            trainbar()

    train_loss = np.mean(train_losses)
    train_time = time.time()
    lr_scheduler.step()

    model.eval()
    val_losses = []
    with torch.no_grad():
        with alive_bar(len(val_examples) // BATCH_SIZE) as evalbar:
            for i in range(0, len(val_examples), BATCH_SIZE):
                examples = val_examples[i:i+BATCH_SIZE]
                inputs = torch.FloatTensor(np.asarray(
                    [example[0] for example in examples])).cuda()
                boxes = torch.FloatTensor(np.asarray(
                    [example[1] for example in examples])).cuda()
                mask = torch.FloatTensor(np.asarray(
                    [example[2] for example in examples])).cuda()
                targets = torch.FloatTensor(np.asarray(
                    [example[3] for example in examples])).cuda()
                loss, outputs = model(inputs,
                                      boxes,
                                      mask,
                                      targets,
                                      MAX_LENGTH=MAX_LENGTH,
                                      BATCH_SIZE=BATCH_SIZE,
                                      NUM_BOXES=NUM_BOXES)
                val_losses.append(loss.item())
                evalbar()

    val_loss = np.mean(val_losses)
    val_loss_track.append(min(val_loss, 2.0))
    val_time = time.time()

    print('RNN-iteration {}: train_time={}, val_time={}, train_loss={}, val_loss={}/{}, bad_loss_streak={}, lr={}'.format(epoch,
          int(train_time - start_time), int(val_time - train_time), train_loss, val_loss, best_loss, bad_loss_streak, optimizer.param_groups[0]['lr']))

    plt.figure(figsize=(8, 7))
    plt.subplot(1, 2, 1)
    plt.plot(loss_track, label='train_loss',
             color=train_loss_color, linewidth=3)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(val_loss_track,
             label=f'val_loss-{round(float(best_loss), 4)}', color=val_loss_color, linewidth=3)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"./figures/train_tracker_loss_{skips[-1]}.jpg",
                dpi=100, bbox_inches='tight')
    plt.close()
    plt.clf()
    plt.cla()

    if best_loss == 9999 or val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), os.path.join(os.path.join(
            args.save_root, "Trackers", args.data_name), 'IOU_{}.pth'.format(skips[-1])))
        bad_loss_streak = 0
    else:
        bad_loss_streak += 1
        if bad_loss_streak > args.max_wait_epochs:
            break
