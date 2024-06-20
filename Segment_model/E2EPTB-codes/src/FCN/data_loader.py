import os
import torch
import numpy as np
import pandas as pd
import random
from PIL import Image
from glob import glob
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from sklearn.utils.class_weight import compute_class_weight

class CervixDataset(Dataset):
    """Cervix Dataset."""

    def __init__(self, images_dir: str = None, masks_dir: str = None, annotations_path: str = None, transform=None):
        """

        Args:
            images_dir: path to directory with images
            masks_dir: path to directory with masks
            annotations_path: path to file with annotations
            transform: image transform [optional]
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.annotations_path = annotations_path
        self.transform = transform
        self.n_class = 2

        self.annotations = pd.read_csv(self.annotations_path, header=None, delimiter=",")
        self.annotations.sort_values
        self.images = glob(self.images_dir + "/*.png")
        self.masks = glob(self.masks_dir + "/*.png")
        self.images.sort()
        self.masks.sort()

        self.class_weights()

    def __getitem__(self, index):
        """Get one sample of data."""
        image = Image.open(self.images[index])
        mask = Image.open(self.masks[index])
        labels = self.__getlabel__(index)

        seed = np.random.randint(2147483647)

        if self.transform is not None:
            random.seed(seed)
            # torch.manual_seed(seed)
            image = self.transform(image)
            random.seed(seed)
            # torch.manual_seed(seed)
            mask = self.transform(mask)
            mask = mask.round()
            
            # _, h, w = mask.shape
            # target = torch.zeros(self.n_class, h, w)
            # for c in range(self.n_class):
            #     target[c, (mask == c).view(h, w)] = 1
            # return image, target, labels
        
        return image, mask, labels

    def __getlabel__(self, idx):
        labels = self.annotations[1][idx]
        if labels in ["CONTROL_1", "CONTROL_2"]:
            labels = 0
        else:
            labels = 1
        
        return labels

    def __len__(self):
        """Get len of data."""
        return len(self.images)

    def class_weights(self):
        label_column = self.annotations.iloc[:, 1]
        no_of_preterm = len(self.annotations[label_column.str.contains('PRETERM')])
        no_of_control = len(self.annotations[label_column.str.contains('CONTROL')])

        preterm = np.ones(no_of_preterm)
        control = np.zeros(no_of_control)
        a = np.append(preterm, control)
        class_weights = compute_class_weight('balanced', np.unique(a), a)

        return class_weights

class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, indices=None, num_samples=None):
        # if indices is not provided, 
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices
            
        # if num_samples is not provided, 
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples
            
        # distribution of classes in the dataset 
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1
                
        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        return dataset.__getlabel__(idx)
                
    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples