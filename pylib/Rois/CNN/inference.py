import os
import cv2
import sys
import json
import torch
import numpy as np
from model import CNNModel
from utils import get_windows_from_bin

def clip(x, lo, hi):
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x

# get smallest area detector size containing this width/height (measured at detector resolution)
def get_detector_size(w, h):
	best_size = None
	for sz in detector_sizes:
		if sz[0] < w or sz[1] < h:
			continue
		if best_size is None or sz[0]*sz[1] < best_size[0]*best_size[1]:
			best_size = sz
	return best_size

def get_windows(images, model):
    with torch.no_grad():
        images = torch.from_numpy(images).cuda()
        images = images.permute(0, 3, 1, 2).float()
        outputs = model(images).cpu().numpy()
    windows = []
    for i in range(len(images)):
        # print(outputs[i, :, :], file=open("outputs.txt", "a"))
        comps = get_windows_from_bin(outputs[i, :, :] > threshold, sizes=sizes)
        l = []
        for comp in comps:
            def transform(x, y):
                x = x*detector_width*32//width
                y = y*detector_height*32//height
                return (x, y)

            # transform component bounds to detector resolution
            # we also need to increase the window bounds to match a detector size
            sx, sy = transform(comp.rect[0], comp.rect[1])
            ex, ey = transform(comp.rect[2]+1, comp.rect[3]+1)
            cx, cy = (sx+ex)//2, (sy+ey)//2
            w, h = get_detector_size(ex-sx, ey-sy)
            cx = clip(cx, w//2, detector_width-w//2)
            cy = clip(cy, h//2, detector_height-h//2)
            bounds = [cx-w//2, cy-h//2, cx+w//2, cy+h//2]

            cells = []
            for x, y in comp.cells:
                sx, sy = transform(x, y)
                ex, ey = transform(x+1, y+1)
                cells.append([sx, sy, ex, ey])

            # bounds = [0, 0, 640, 352]
            l.append({
                'bounds': bounds,
                'cells': cells,
            })
        windows.append(l)
    return windows

batch_size = int(sys.argv[1])
width = int(sys.argv[2])
height = int(sys.argv[3])
threshold = float(sys.argv[4])
detector_width = int(sys.argv[5])
detector_height = int(sys.argv[6])
detector_sizes = json.loads(sys.argv[7])
model_path = sys.argv[8]
device_id = sys.argv[9]

os.environ['CUDA_VISIBLE_DEVICES'] = f'{device_id}'

# compute our sizes by rounding down from detector sizes
sizes = []
for w, h in detector_sizes:
	w = w*width//detector_width//32
	h = h*height//detector_height//32
	sizes.append((w, h))
sizes.append((width//32, height//32))

model = CNNModel()
model.load_state_dict(torch.load(model_path))
model.cuda()
model.eval()

seen = 0
stdin = sys.stdin.detach()
while True:
    buf = stdin.read(batch_size*width*height*3)
    if not buf:
        break
    ims = np.frombuffer(buf, dtype='uint8').reshape((batch_size, height, width, 3))
    # imr = np.copy(cv2.cvtColor(ims[0], cv2.COLOR_RGB2BGR))
    # cv2.imwrite(f'test/{seen}.png', imr)
    ims = np.copy(ims)
    ims_norm = []
    for im in ims:
        ims_norm.append((im.astype(np.float32) / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225])
    ims_norm = np.array(ims_norm)
    windows = get_windows(ims_norm, model)
    # print('windows', windows, file=open('roi_sizes.txt', 'a'))
    print('json'+json.dumps(windows), flush=True)
    seen += 1