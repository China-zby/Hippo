import os
import cv2
import sys
import torch
import random
import argparse
import numpy as np
from dataloader import *
from model import CNNModel
from alive_progress import alive_bar

def train(dataloader, criterion, optimizer):
    model.train()
    losses = []
    with alive_bar(len(dataloader), title="Train") as bar:
        for images, labels in dataloader:
            images = images.cuda()
            labels = labels.cuda()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            bar()
    return losses
        
def val(dataloader, criterion):
    model.eval()
    losses, presult, nresult = [], [], []
    with torch.no_grad():
        with alive_bar(len(dataloader), title="Val") as bar:
            for images, labels in dataloader:
                images = images.cuda()
                labels = labels.cuda()
                outputs = model(images)
                loss = criterion(outputs, labels)
                # print(f"loss: {loss.item()}")
                losses.append(loss.item())
                # print(f"outputs: {outputs} labels: {labels}")
                
                for pred, label in zip(outputs, labels):
                    pred = torch.argmax(pred).cpu().numpy()
                    label = label.cpu().numpy()
                    # print(f"pred: {pred}, label: {label}")
                    if label == 1:
                        presult.append(1 if pred > 0.5 else 0)
                    else:
                        # print(f"pred: {pred}, label: {label}")
                        nresult.append(1 if pred < 0.5 else 0)
                bar()

    return np.mean(losses), np.mean(presult), np.mean(nresult)

parses = argparse.ArgumentParser()
parses.add_argument('--data_dir', type=str, default='/home/xyc/otif/otif-dataset/dataset', help='Directory for storing data')
parses.add_argument('--data_name', type=str, default='amsterdam', help='Name of the dataset')
parses.add_argument('--width', type=int, default=640, help='Width of the image')
parses.add_argument('--height', type=int, default=352, help='Height of the image')
parses.add_argument('--save_dir', type=str, default='./weights', help='Save directory for the model')
parses.add_argument('--epochs', type=int, default=9999, help='Number of epochs to train')
parses.add_argument('--max_wait_epochs', type=int, default=25, help='Number of epochs to train')
parses.add_argument('--batch_size', type=int, default=32, help='Batch size')
parses.add_argument('--classes', type=int, default=2, help='classes')
parses.add_argument('--hidden_size', type=int, default=128, help='hidden_size')
parses.add_argument('--filters', type=int, default=32, help='nb_filters')
parses.add_argument('--layers', type=int, default=1, help='nb_layers')
parses.add_argument('--visual_label', action='store_true', help='visual_label')

args = parses.parse_args()

human = True
dataset_name = args.data_name
epochs = args.epochs
BatchSize = args.batch_size
train_width = args.width
train_height = args.height
visual_label = args.visual_label

print(f'''dataset_name: {dataset_name}, 
          epochs: {epochs}, BatchSize: {BatchSize}, 
          train_width: {train_width}, train_height: {train_height}''')

seed = 2023
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

data_root = os.path.join(args.data_dir, args.data_name, "train/seg-train") # f"/home/xyc/otif/otif-dataset/dataset/{dataset_name}/train/seg-train"
new_label_dir = os.path.join(data_root, "images")

if not os.path.exists(new_label_dir):
    os.makedirs(new_label_dir)
    
labeled_image_dir = os.path.join(data_root, "labeledimages")
if not os.path.exists(labeled_image_dir):
    os.makedirs(labeled_image_dir)

imagepaths = os.listdir(os.path.join(data_root, "images"))
imagepaths = [imagepath for imagepath in imagepaths if imagepath.endswith(".jpg")]

trainImageSize = (train_height, train_width)
# train_transform = build_transform("train", trainImageSize)
# val_transform = build_transform("val", trainImageSize)

Rawdataset = RawDataset(data_root)

if visual_label: Rawdataset.visual_label()

datalength = len(Rawdataset.dataset)
# trainIndexs = [i for i in range(datalength) if i % 10 != 0]
# valIndexs = [i for i in range(datalength) if i % 10 == 0]
posIndexs, negIndexs = [], []
for i in range(datalength):
    if Rawdataset.dataset[i][1] == 1: posIndexs.append(i)
    else: negIndexs.append(i)

trainIndexs = random.sample(posIndexs, int(len(posIndexs) * 0.8)) + random.sample(negIndexs, int(len(negIndexs) * 0.8))
valIndexs = [i for i in range(datalength) if i not in trainIndexs]

print(f"trainIndexs: {len(trainIndexs)}, valIndexs: {len(valIndexs)}")
print(f"posIndexs: {len(posIndexs)}, negIndexs: {len(negIndexs)}")
print(f"train posInexs: {len([i for i in trainIndexs if i in posIndexs])}, train negIndexs: {len([i for i in trainIndexs if i in negIndexs])}")
print(f"valid posInexs: {len([i for i in valIndexs if i in posIndexs])}, valid negIndexs: {len([i for i in valIndexs if i in negIndexs])}")

trainDataset = MyDataset([Rawdataset.dataset[i] for i in trainIndexs], [train_width, train_height]) # , transform=train_transform
valDataset = MyDataset([Rawdataset.dataset[i] for i in valIndexs], [train_width, train_height]) # , transform=val_transform
trainDataloader = MyDataloader(trainDataset, batch_size=BatchSize, shuffle=True, num_workers=4)
valDataloader = MyDataloader(valDataset, batch_size=BatchSize * 4, shuffle=False, num_workers=4)

model = CNNModel([3, args.width, args.height],
                  args.classes, args.hidden_size,
                  args.filters, args.layers)
model.cuda()

label_weight = torch.tensor([Rawdataset.posNum / (Rawdataset.posNum + Rawdataset.negNum),
                             Rawdataset.negNum / (Rawdataset.posNum + Rawdataset.negNum)]).cuda()
print(f"label_weight: {label_weight}")

criterion = torch.nn.CrossEntropyLoss(weight=label_weight) # 198 1368
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs // 50, eta_min=0.000001)

decay_epoch_number = 0
best_loss, best_f1 = 9999, -9999

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    print("-" * 10)
    losses = train(trainDataloader, criterion, optimizer)
    val_loss, pacc, nacc = val(valDataloader, criterion)
    print(f"train loss: {np.mean(losses):.4f}, val loss: {val_loss:.4f}, pacc: {pacc:.4f}, nacc: {nacc:.4f}, f1: {2 * pacc * nacc / (pacc + nacc):.4f}, lr: {optimizer.param_groups[0]['lr']:.6f}, best f1: {best_f1:.4f}, decay epoch number: {decay_epoch_number}")

    f1 = 2 * pacc * nacc / (pacc + nacc)
    
    scheduler.step(epoch)
    
    if f1 > best_f1:
        best_f1 = f1
        
        if not os.path.exists(f"./weights/Filters/{dataset_name}"):
            os.makedirs(f"./weights/Filters/{dataset_name}")
        
        torch.save(model.state_dict(), f"./weights/Filters/{dataset_name}/CNN_{train_width}_{train_height}.pth")
    else:
        decay_epoch_number += 1
        if decay_epoch_number > args.max_wait_epochs:
            break