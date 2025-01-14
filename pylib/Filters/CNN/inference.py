import os
import cv2
import sys
import time
import json
import torch
import numpy as np
from model import CNNModel

BatchSize = int(sys.argv[1])
InputImageWidth = int(sys.argv[2])
InputImageHeight = int(sys.argv[3])
ImageWidth = int(sys.argv[4])
ImageHeight = int(sys.argv[5])
Threshold = float(sys.argv[6])
weightPath = sys.argv[7]
device_id = sys.argv[8]

os.environ['CUDA_VISIBLE_DEVICES'] = f'{device_id}'

model = CNNModel([3, InputImageWidth, InputImageHeight],
                 2, 128,
                 32, 1)
model.load_state_dict(torch.load(weightPath))
model.cuda()
model.eval()

stdin = sys.stdin.detach()
while True:
    buf = stdin.read(BatchSize*ImageWidth*ImageHeight*3)
    if not buf:
        break
    ims = np.frombuffer(buf, dtype='uint8').reshape(
        (BatchSize, ImageHeight, ImageWidth, 3))
    # print(ims.shape, file=open('ims.txt', 'a'))
    process_ims = []
    for im in ims:
        im = cv2.resize(im, (InputImageWidth, InputImageHeight))
        im = im.astype(np.float32) / 255.0
        im = (im - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        process_ims.append(im)
    ims = np.array(process_ims)
    ims = torch.FloatTensor(ims).permute(0, 3, 1, 2).cuda()
    # inference
    with torch.no_grad():
        t0 = time.time()
        out = model(ims)  # (batch_size, 2)
        out = torch.softmax(out, dim=1)[:, 1]  # (batch_size)
        t1 = time.time()
        print(f"latency: {t1-t0}", "Result: {out}",
              file=open('latency.txt', 'a'))
    # convert to numpy
    out = out.cpu().numpy()
    # convert to binary
    out = (out > Threshold).astype(np.uint8).tolist()
    # print(out, file=open('out.txt', 'a'))
    # convert to bytes
    # out = out.tobytes()
    # write to stdout
    print('json'+json.dumps(out), flush=True)
