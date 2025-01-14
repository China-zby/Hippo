import os
import cv2
import json
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# def build_transform(type, image_size=[384, 640]):
#     if type == "train":
#         outputTrans = transforms.Compose([
#             transforms.RandomResizedCrop(image_size),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         ])
#     else:
#         outputTrans = transforms.Compose([
#             transforms.Resize(image_size),
#             transforms.CenterCrop(image_size),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         ])
#     return outputTrans

def read_image(image_path):
    try:
        image = Image.open(image_path)
    except:
        print("Error opening image: {}".format(image_path))
        return None
    return image

class RawDataset():
    def __init__(self, data_root, 
                 target_classes=["car", "bus", "truck"]):
        self.imageDir = os.path.join(data_root, "images")
        self.labelDir = os.path.join(data_root, "images")
        self.target_classes = target_classes
        self.filter_target = True if self.target_classes is not None else False
        
        self.dataset = self.process_dataset()
        
    def process_dataset(self):
        imageNames = [imagepath for imagepath in os.listdir(self.imageDir) if imagepath.endswith(".jpg")]
        images = [os.path.join(self.imageDir, imagepath) for imagepath in imageNames]
        labels = [os.path.join(self.labelDir, imagepath.replace(".jpg", ".json")) for imagepath in imageNames]
        dataset = []
        posNum, negNum = 0, 0
        for image, label in zip(images, labels):
            labeldata = json.load(open(label, "r"))
            if self.filter_target:
                labeldata = [item_data for item_data in labeldata if item_data["class"] in self.target_classes]
            if len(labeldata) == 0:
                dataset.append((image, 0, labeldata))
                negNum += 1
            else:
                posNum += 1
                dataset.append((image, 1, labeldata))
        print("Positive: {}, Negative: {}".format(posNum, negNum))
        self.posNum = posNum
        self.negNum = negNum
        return dataset
    
    def visual_label(self):
        from tqdm import tqdm
        print("Visualizing label...")
        for image, label, labeldata in tqdm(self.dataset):
            imageRaw = cv2.imread(image)
            for item_data in labeldata:
                left, top, right, bottom = item_data["left"], item_data["top"], item_data["right"], item_data["bottom"]
                cv2.rectangle(imageRaw, 
                              (left, top),
                              (right, bottom),
                              (0, 255, 0), 2)
                cv2.putText(imageRaw, "%.2f" % (float(item_data['score'])),
                            (left, top), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(imageRaw, "Label: {}".format(label), fontScale=3, org=(100, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(0, 0, 255), thickness=2)
            cv2.imwrite(image.replace("images", "labeledimages"), imageRaw)
        print("Done!")

class MyDataset():
    def __init__(self, dataset, input_dim, transform=None):
        self.transform = transform
        self.input_dim = input_dim
        self.dataset = dataset
        
    def __getitem__(self, index):
        imagePath, label, _ = self.dataset[index]
        if self.transform is not None:
            imageRaw = read_image(imagePath)
            image = self.transform(imageRaw)
        else:
            imageRaw = cv2.imread(imagePath)
            imageRaw = cv2.cvtColor(imageRaw, cv2.COLOR_BGR2RGB)
            resized_im = cv2.resize(imageRaw, (self.input_dim[0], self.input_dim[1])).astype(np.float32) / 255.0
            normalize_image = (resized_im - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
            image = torch.FloatTensor(normalize_image.transpose(2, 0, 1))
        return image, label
    
    def __len__(self):
        return len(self.dataset)
    
class MyDataloader():
    def __init__(self, dataset, batch_size, shuffle, num_workers):
        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    def __iter__(self):
        return iter(self.dataloader)
    
    def __len__(self):
        return len(self.dataloader)