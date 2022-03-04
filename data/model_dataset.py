import os
from PIL import Image
import torch
from torch.utils.data import Dataset

class FontDataset(Dataset):
    def __init__(self, image_root, image_1, image_2, label, transform, scale = 1):
        self.image_root = image_root
        self.image_1 = image_1
        self.image_1 = image_2
        self.label = label
        self.scale = scale
        self.transform = transform

    def __len__(self):
        return int(len(self.label) * self.scale)

    def __getitem__(self, index):
        index = index % len(self.label)
        img_path_1 = os.path.join(self.image_root, self.image_1[index])
        img_path_2 = os.path.join(self.image_root, self.image_2[index])
        
        img_1 = Image.open(img_path_1).convert('RGB')
        img_2 = Image.open(img_path_2).convert('RGB')
        return self.transform(img_1), self.transform(img_2), self.label[index]

