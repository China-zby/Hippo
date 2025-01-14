import os
import time
import torch
import argparse
import numpy as np
from loguru import logger
from model import CNNModel
from alive_progress import alive_bar
from torch.utils.data import Dataset
from dataloader import LoadDatasets, CustomDataset

if __name__ == '__main__':
    parses = argparse.ArgumentParser()
    parses.add_argument('--data_dir', type=str, default='/home/xyc/otif/otif-dataset/dataset', help='Directory for storing data')
    parses.add_argument('--data_name', type=str, default='amsterdam', help='Name of the dataset')
    parses.add_argument('--width', type=int, default=640, help='Width of the image')
    parses.add_argument('--height', type=int, default=352, help='Height of the image')
    parses.add_argument('--save_dir', type=str, default='./weights', help='Save directory for the model')
    parses.add_argument('--epochs', type=int, default=9999, help='Number of epochs to train')
    parses.add_argument('--max_wait_epochs', type=int, default=25, help='Number of epochs to train')
    parses.add_argument('--batch_size', type=int, default=32, help='Batch size')
    
    args = parses.parse_args()
    
    save_dir = os.path.join(args.save_dir, 'Rois', args.data_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    classNames = ['car', 'bus', 'truck']    
    # make dataloader for training
    RawDataset = LoadDatasets(os.path.join(args.data_dir, args.data_name), classNames, (args.width, args.height))
    
    val_dataset, test_dataset, train_dataset = CustomDataset(RawDataset.val_dataset), \
                                               CustomDataset(RawDataset.test_dataset), \
                                               CustomDataset(RawDataset.train_dataset)

    logger.info('Train dataset size: {}'.format(len(train_dataset)))
    logger.info('Val dataset size: {}'.format(len(val_dataset)))
    logger.info('Test dataset size: {}'.format(len(test_dataset)))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # make model and train
    best_loss = None
    bad_loss_streak = 0
    model = CNNModel().cuda()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCELoss()
    
    for epoch in range(args.epochs):
        # print('Epoch: {}'.format(epoch))
        start_time = time.time()
        train_losses = []
        with alive_bar(len(train_loader), title=f"Train Epoch: {epoch}") as bar:
            for batch_idx, (images, targets, _) in enumerate(train_loader):
                images, targets = images.cuda(), targets.cuda()
                # print(images.shape, targets.shape)
                images = images.permute(0, 3, 1, 2).float()
                # print(images.shape, targets.shape)
                optimizer.zero_grad()
                output = model(images)
                loss = criterion(output, targets)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
                bar()
        train_loss = np.mean(train_losses)
        train_time = time.time()
            
        val_losses = []
        with alive_bar(len(val_loader), title=f"Val Epoch: {epoch}") as bar:
            with torch.no_grad():
                for batch_idx, (images, targets, _) in enumerate(val_loader):
                    images, targets = images.cuda(), targets.cuda()
                    # print(images.shape, targets.shape)
                    images = images.permute(0, 3, 1, 2).float()
                    # print(images.shape, targets.shape)
                    output = model(images)
                    loss = criterion(output, targets)
                    val_losses.append(loss.item())
                    bar()
        val_loss = np.mean(val_losses)
        val_time = time.time()
        
        logger.info(f'iteration ({args.width}, {args.height}) {epoch}: train_time={int(train_time - start_time)}, val_time={int(val_time - train_time)}, train_loss={train_loss}, val_loss={val_loss}/{best_loss}, bad_loss_streak={bad_loss_streak}')
        
        if best_loss is None or val_loss < best_loss:
            best_loss = val_loss
            bad_loss_streak = 0
            torch.save(model.state_dict(), os.path.join(save_dir, f'CNN_{args.width}_{args.height}.pth'))
        else:
            bad_loss_streak += 1
            if bad_loss_streak > args.max_wait_epochs:
                break