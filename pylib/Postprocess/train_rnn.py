import os
import random
import argparse

import torch
import numpy as np
from dataset import TrackDataset
from alive_progress import alive_bar
from torch.utils.data import DataLoader, random_split

from model import PSRNN
from utils import visual_gtandpred_track_data

if __name__ == "__main__":
    parse = argparse.ArgumentParser("Run query metrics")
    parse.add_argument("--data_root", default="./", help="data root")
    parse.add_argument("--data_name", default="amsterdam", help="test mode")
    parse.add_argument("--mode", default="train", help="train mode")
    args = parse.parse_args()

    scene2id = {
        "amsterdam": 0,
        "warsaw":    1, "shibuya": 2,
        "jackson": 3, "caldot1": 4,
        "caldot2": 5, "uav": 6,
    }

    seed = 2023
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    split_ratio = 0.9

    data_dir = os.path.join(args.data_root, args.data_name, args.mode)
    tracks = os.listdir(os.path.join(data_dir, "tracks"))
    tracks = list(filter(lambda x: x.endswith(".json"), tracks))

    # plus all tracks
    tracks += os.listdir(os.path.join(os.path.join(args.data_root,
                         args.data_name, "streamline"), "tracks"))
    tracks = list(filter(lambda x: x.endswith(".json"), tracks))

    model = PSRNN(scene2id)
    model = model.cuda()
    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=10, gamma=0.8)

    dataset = TrackDataset(os.path.join(data_dir, "tracks"), tracks, scene2id)
    train_size = int(split_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size])

    print(f"train size: {len(train_dataset)} test size: {len(test_dataset)}")

    train_dataloader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=8)
    test_dataloader = DataLoader(
        test_dataset, batch_size=128, shuffle=False, num_workers=8)

    best_loss = 9999.9

    for epoch in range(1000):
        loss_epoch = []
        model.train()
        with alive_bar(len(train_dataloader)) as bar:
            for batch in train_dataloader:
                trajectory, prefix, suffix, scene_id, trajectory_lengths = batch

                trajectory = trajectory.cuda()
                prefix = prefix.cuda()
                suffix = suffix.cuda()
                scene_id = scene_id.cuda()
                pred_prefix, pred_suffix = model(
                    trajectory, trajectory_lengths, scene_id)

                loss = loss_func(pred_prefix, prefix) + \
                    loss_func(pred_suffix, suffix)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_epoch.append(loss.item())

                bar()
        print(
            f"epoch: {epoch} loss: {sum(loss_epoch) / len(loss_epoch)} lr: {optimizer.param_groups[0]['lr']:.6f}")

        if epoch < 500:
            lr_scheduler.step()

        # test
        model.eval()
        with torch.no_grad():
            loss_epoch = []
            data_id = 0
            for batch in test_dataloader:
                trajectory, prefix, suffix, scene_id, trajectory_lengths = batch

                trajectory = trajectory.cuda()
                prefix = prefix.cuda()
                suffix = suffix.cuda()
                scene_id = scene_id.cuda()
                pred_prefix, pred_suffix = model(
                    trajectory, trajectory_lengths, scene_id)

                loss = loss_func(pred_prefix, prefix) + \
                    loss_func(pred_suffix, suffix)
                loss_epoch.append(loss.item())

                # visual trajectory and prefix suffix
                trajectory = trajectory.cpu().numpy()
                pred_prefix = pred_prefix.cpu().numpy()
                pred_suffix = pred_suffix.cpu().numpy()
                scene_id = scene_id.cpu().numpy()
                trajectory_lengths = trajectory_lengths.cpu().numpy()
                prefix = prefix.cpu().numpy()
                suffix = suffix.cpu().numpy()
                for traj, pred_pf, pred_sf, sid, sl, pf, sf in zip(trajectory, pred_prefix, pred_suffix, scene_id, trajectory_lengths, prefix, suffix):
                    _, scene_id, width, height, _ = test_dataset.dataset.dataset[
                        data_id]
                    traj = traj[:sl]
                    track_data = [[int(p[0] * width), int(p[1] * height),
                                   int(p[2] * width), int(p[3] * height),
                                   int(p[4] * dataset.frame_norm)] for p in traj]
                    pred_pf = [int(pred_pf[0] * width), int(pred_pf[1] * height),
                               int(pred_pf[2] *
                                   width), int(pred_pf[3] * height),
                               int(pred_pf[4] * dataset.frame_norm)]
                    pred_sf = [int(pred_sf[0] * width), int(pred_sf[1] * height),
                               int(pred_sf[2] *
                                   width), int(pred_sf[3] * height),
                               int(pred_sf[4] * dataset.frame_norm)]
                    pf = [int(pf[0] * width), int(pf[1] * height),
                          int(pf[2] * width), int(pf[3] * height),
                          int(pf[4] * dataset.frame_norm)]
                    sf = [int(sf[0] * width), int(sf[1] * height),
                          int(sf[2] * width), int(sf[3] * height),
                          int(sf[4] * dataset.frame_norm)]

                    visual_gtandpred_track_data(track_data, pred_pf, pred_sf, pf, sf,
                                                width, height, dataset.frame_norm, data_id, dataset.id2scene[sid.item()])
                    data_id += 1
            print(
                f"epoch: {epoch} Eval loss: {sum(loss_epoch) / len(loss_epoch)}")

            mean_loss = sum(loss_epoch) / len(loss_epoch)
            if mean_loss < best_loss:
                best_loss = mean_loss
                torch.save(model.state_dict(), "./weights/Postprocess/rnn.pth")
                print(f"best loss: {best_loss}")
