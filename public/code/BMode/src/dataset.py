from torch.utils.data import Dataset
import os
import random
import cv2
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms

class Resize(object):

    def __init__(self, output_size: tuple):
        self.output_size = output_size

    def __call__(self, sample):
        # 图像
        image = sample['image']
        # 使用skitimage.transform对图像进行缩放
        image_new = cv2.resize(image, self.output_size)
        return {'image': image_new, 'label': sample['label']}

class ToGray(object):
    def __call__(self, sample):
        image = sample['image']
        image_new = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # image_new = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return {'image': image_new, 'label': sample['label']}

class ToTensor(object):
    def __call__(self, sample):
        image = np.array(sample['image'])
        # image_new = np.transpose(image, (2, 0, 1))
        tensor_image =  torch.from_numpy(image).float()

        return {'image':tensor_image.unsqueeze(0),
                'label': sample['label']}

class RandomCrop(object):
    def __call__(self,sample):
        image = sample['image']
        pil_image =  Image.fromarray(image)
        trans = transforms.RandomCrop(256, padding=4)
        new_image = trans(pil_image)
        return {'image': new_image,
                'label': sample['label']}

class RandomHorizontalFlip(object):
    def __call__(self, sample):
        image = sample['image']
        trans = transforms.RandomHorizontalFlip()
        new_image = trans(image)
        return {'image': new_image,
                'label': sample['label']}



class MyDataset(Dataset):
    def __init__(self,healthy_dir, sick_dir, transform = None):
        name_dict = dict()
        for name in os.listdir(healthy_dir):
            path_name = healthy_dir + name
            name_dict[path_name] = 0
        for name in os.listdir(sick_dir):
            path_name = sick_dir + name
            name_dict[path_name] = 1
        keys = list(name_dict.keys())
        random.shuffle(keys)
        self.name_list =  [(key, name_dict[key]) for key in keys]
        self.transform = transform

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        image_path = self.name_list[idx][0]
        label = self.name_list[idx][1]
        image = cv2.imread(image_path)
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample


class FeatureDataset(Dataset):
    def __init__(self,healthy_dir, sick_dir, transform = None):
        name_dict = dict()
        for name in os.listdir(healthy_dir):
            path_name = healthy_dir + name
            name_dict[path_name] = 0
        for name in os.listdir(sick_dir):
            path_name = sick_dir + name
            name_dict[path_name] = 1
        keys = list(name_dict.keys())
        random.shuffle(keys)
        self.name_list =  [(key, name_dict[key]) for key in keys]
        self.transform = transform

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        image_path = self.name_list[idx][0]
        label = self.name_list[idx][1]
        image = cv2.imread(image_path)
        sample = {'image': image, 'label': label, 'path':image_path}
        if self.transform:
            sample = self.transform(sample)
        return sample





