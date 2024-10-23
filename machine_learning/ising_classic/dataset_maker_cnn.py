from torchvision.io.image import read_image
from torchvision import transforms as torch_transform
from torch.utils.data import Dataset
from PIL import Image
import torch
import pandas as pd
import os


class ImageDataset(Dataset):
    def __init__(self,
                 images_path,
                 annotations_path,
                 transform=None,
                 target_transform=None):
        self.img_labels = pd.read_csv(annotations_path)
        self.img_dir = images_path
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path)
        image = torch_transform.ToTensor()(image)
        label = self.img_labels.iloc[idx, 1]
        # print(label)
        label = torch.nn.functional.one_hot(torch.tensor(label), num_classes=2).type(torch.FloatTensor)
        # print(label)

        temp = self.img_labels.iloc[idx, 2]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label, temp
