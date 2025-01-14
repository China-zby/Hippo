import os
import cv2
import utils
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset

class LoadDatasets(object):
    def __init__(self, data_path, classnames, input_dim):
        image_path = os.path.join(data_path, "train/seg-train/images")
        self.train_dataset, self.val_dataset, self.test_dataset = self.load_images(image_path, classnames, input_dim)
    
    def load_images(self, data_path, classnames, input_dim):
        train_data, val_data, test_data = [], [], []
        fnames = [fname for fname in os.listdir(data_path) if fname.endswith('.jpg')]
        for i, fname in enumerate(tqdm(fnames)):
            label = fname.split('.jpg')[0]
            # im = skimage.io.imread(os.path.join(data_path, label+'.jpg'))
            # resized_im = skimage.transform.resize(im, [input_dim[1], input_dim[0]], preserve_range=True).astype('uint8')
            
            im = cv2.imread(os.path.join(data_path, label+'.jpg'))
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            resized_im = cv2.resize(im, (input_dim[0], input_dim[1])) 
            normalize_im = (resized_im.astype(np.float32) / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
            target = utils.load_target(os.path.join(data_path, label+'.json'), classnames, input_dim, (im.shape[1], im.shape[0]), lenient=True)
            label = int(label)
            # print(resized_im.shape)
            if label % 4 == 0:
                val_data.append((normalize_im, target, int(label)))
            elif label % 4 == 1:
                test_data.append((normalize_im, target, int(label)))
            else:
                train_data.append((normalize_im, target, int(label)))
            
        return train_data, val_data, test_data

class CustomDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, target, label = self.dataset[idx]
        if self.transform:
            image = self.transform(image)
        return image, target, label