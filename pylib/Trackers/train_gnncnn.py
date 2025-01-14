import os
import json
import torch
import argparse
import numpy as np
from time import time
import torch.optim as optim
from torch.utils.data import DataLoader

from model import MOTGraphModel
from dataloader import MOTGraphDataset

def refresh():
    """Clears the console and refreshes the screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

parses = argparse.ArgumentParser()
parses.add_argument('--data_root', type=str, default='/home/xyc/otif/otif-dataset/dataset', help='path to dataset')
parses.add_argument('--save_root', type=str, default='/home/lzp/go-work/src/otifpipeline/weights', help='path to dataset')
parses.add_argument('--data_name', type=str, default='amsterdam', help='path to dataset')
parses.add_argument('--data_flag', type=str, default='tracker', help='the flag of the dataset')
parses.add_argument('--cfg_path', type=str, default="/home/lzp/go-work/src/otifpipeline/pylib/Trackers/GNNCNN/cfg_128.json", help='Number of epochs to train')
parses.add_argument('--width', type=int, default=1280, help='the width of the input image')
parses.add_argument('--height', type=int, default=720, help='the height of the input image')
parses.add_argument('--crop_size', type=int, default=64, help='the size of the cropped image')
parses.add_argument('--max_wait_epochs', type=int, default=25, help='Number of epochs to train')
parses.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate to train')
args = parses.parse_args()

root_path = os.path.join(args.data_root, args.data_name)
with open(os.path.join(args.cfg_path), 'r') as f:
    cfg = json.load(f)
    skips = cfg['Freqs']
    # fps = cfg['FPS']
    # MAX_LENGTH = cfg['Max_length']
    # NORM = cfg['Norm']
    # NUM_BOXES = cfg['Num_boxes']
    # BATCH_SIZE = cfg['Batch_size']

learning_rate = args.learning_rate

# Define parameters
crop_size = args.crop_size
width, height = args.width, args.height
dataset = f'{args.data_root}/{args.data_name}/{args.data_flag}'
batch_size = MOTGraphModel.BATCH_SIZE
num_epochs = 30
train_val_split = 0.9

# Create dataset
mot_graph_dataset = MOTGraphDataset(os.path.join(args.data_root, args.data_name, args.data_flag), skips,
                                    width, height, crop_size)

# Split dataset into training and validation sets
num_train = int(len(mot_graph_dataset) * train_val_split)
num_val = len(mot_graph_dataset) - num_train
train_dataset, val_dataset = torch.utils.data.random_split(mot_graph_dataset, [num_train, num_val])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=mot_graph_dataset.collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=mot_graph_dataset.collate_fn)

# Initialize the model and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MOTGraphModel(8, 6).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
best_val_loss = float("inf")
epochs_without_better = 0
train_idx = 0

# stdscr = curses.initscr()
# stdscr.clear()  # Clear the screen
# stdscr.refresh()  # Refresh the screen

# stdscr.addstr(0, 0, 'begin training')

total_train_losses, total_valid_losses = [], []
for epoch in range(num_epochs):
    model.train()
    train_losses = []

    # with alive_bar(len(train_loader)) as bar:
    time_marker = time()
    for input_graph, input_crops in train_loader:
        time_start = time()
        input_graph = input_graph.to(device)
        input_crops = input_crops.to(device)

        optimizer.zero_grad()
        outputs = model(input_graph, input_crops)
        loss = model.loss(outputs, input_graph.y)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
        
        time_end = time()
        
        if train_idx % 10 == 0:
            time_skip = (time_end - time_marker) / (train_idx + 1)
            time_total = time_skip * len(train_loader)
            time_left = time_total - time_skip * (train_idx + 1)
            time_total_m, time_total_s = divmod(time_total, 60)
            time_left_m, time_left_s = divmod(time_left, 60)
            print(f'\r Epoch {epoch + 1}, Items {(train_idx + 1)}/{len(train_loader)}, Times [{int(time_left_m):02}:{int(time_left_s):02}]/[{int(time_total_m):02}:{int(time_total_s):02}], Train Loss: {np.mean(train_losses):.4f}', end='')
        
        train_idx += 1
    train_loss = sum(train_losses) / len(train_losses)
    total_train_losses.append(train_loss)

    # Validation
    model.eval()
    val_losses = []

    with torch.no_grad():
        for input_graph, input_crops in val_loader:
            input_graph = input_graph.to(device)
            input_crops = input_crops.to(device)
        
            outputs = model(input_graph, input_crops)
            loss = model.loss(outputs, input_graph.y)

            val_losses.append(loss.item())
                
    val_loss = sum(val_losses) / len(val_losses)
    total_valid_losses.append(val_loss)

    print(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    # stdscr.addstr(1, 0, f'GNNCNN SKIP-{skips[-1]} Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    # stdscr.refresh()  # Refresh the screen

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), os.path.join(args.save_root, "Trackers", f'{args.data_name}/GNNCNN_{skips[-1]}.pth'))
        epochs_without_better = 0
    else:
        epochs_without_better += 1
        if epochs_without_better >= args.max_wait_epochs // 10:
            if learning_rate < 1e-5:
                break

            print('Reducing learning rate to 1e-4')
            learning_rate *= 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate
            epochs_without_better = 0

# curses.endwin()

# # visualize
# import matplotlib.pyplot as plt
# plt.plot(total_train_losses, label='train')
# plt.plot(total_valid_losses, label='valid')
# plt.legend()
# plt.savefig('loss_GNNCNN.png')