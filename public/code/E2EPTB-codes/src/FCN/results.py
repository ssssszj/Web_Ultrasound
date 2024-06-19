import numpy as np
import cv2
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.utils import make_grid
from data_loader import CervixDataset
from PIL import Image
from utils import get_samples_and_mask_for_label

cervix_dataset = CervixDataset("../../data/Bezier", "../../data/Beziermask", "../../data/annotations_final.csv")
fcn = torch.load('../models/unet_.pt', map_location='cpu')
print(fcn)
size = (700, 500)

def drawContourOnImage(image, mask, color=(255,255,255)):
    opencvMask = np.array(mask)
    _, thresh = cv2.threshold(opencvMask, 127, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contour = contours[0]
    for i in range(len(contours)):
        c = contours[i]
        if len(c) > len(contour):
            contour = c

    opencvImage = cv2.cvtColor(np.array(image).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.drawContours(opencvImage, [contour], -1, color, 2)

    return Image.fromarray(opencvImage).resize(size)

def preterm_result_grid(number_of_images=8, label=0):
    transform = transforms.Compose([
        transforms.Resize(size=(256, 256)),
        transforms.ToTensor()
    ])
    
    images = get_samples_and_mask_for_label(cervix_dataset, num_of_samples=number_of_images, label=label)
    imglist = list()
    for (image, mask) in images:
        img = transform(image).unsqueeze(0)
        pred_mask = fcn(Variable(img))
        data = pred_mask.squeeze(0).cpu().data
        pred_mask = transforms.ToPILImage()(data).resize(size)
        # img = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(img)
        # img = transforms.ToPILImage()(img)
        combined_image = drawContourOnImage(image, mask)
        combined_image = drawContourOnImage(combined_image, pred_mask, color=(0,255,0))
        imglist.append(transforms.ToTensor()(combined_image))
    grid = make_grid(imglist, normalize=False, nrow=4)
    grid = transforms.ToPILImage()(grid)
    grid.show()


if __name__ == "__main__":
    preterm_result_grid()
