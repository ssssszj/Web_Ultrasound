import sys
sys.path.append('../src')

import numpy as np
import cv2
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from data_loader import CervixDataset
from PIL import Image
from sklearn.model_selection import train_test_split
from utils import get_samples_for_label

cervix_dataset = CervixDataset("../../data/Bezier", "../../data/Beziermask", "../../data/annotations_final.csv")

unet = torch.load('../../models/unet_.pt', map_location='cpu')

def extract_cervix(image, model):
    orginal_shape = image.size
    image = image.resize((256, 256))
    inputs = Variable(transforms.ToTensor()(image).unsqueeze(0))
    pred_mask, _ = model(inputs)
    outputs = pred_mask.round().squeeze(0).cpu().data
    mask = transforms.ToPILImage()(outputs)
    background = Image.new('RGB', (256, 256), color='black')

    return mask.resize(orginal_shape), Image.composite(image, background, mask).resize(orginal_shape)

def drawContourOnImage(image, mask):
    opencvMask = np.array(mask)
    _, thresh = cv2.threshold(opencvMask, 127, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contour = contours[0]
    for i in range(len(contours)):
        c = contours[i]
        if len(c) > len(contour):
            contour = c

    opencvImage = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    cv2.drawContours(opencvImage, [contour], -1, (0,255,0), 8)

    return Image.fromarray(opencvImage)

_, test_indices = train_test_split(np.arange(len(cervix_dataset)), test_size=0.2, random_state=42)

images = [cervix_dataset[i][0] for i in test_indices[4:8]]
masks = [cervix_dataset[i][1] for i in test_indices[4:8]]

fig, axis = plt.subplots(len(images), 4, figsize=(10, 10), sharex=True, sharey=True)
axis[0][0].set_title("Input")
axis[0][1].set_title("GD")
axis[0][2].set_title("Unet mask")
axis[0][3].set_title("Result")
for image, mask, (ax1, ax2, ax3, ax4) in zip(images, masks, axis):
    umask, cervix = extract_cervix(image, unet)
    ax1.imshow(image)
    _mask = transforms.ToTensor()(mask).round()
    mask = transforms.ToPILImage()(_mask)
    ax2.imshow(mask, cmap='gray')
    ax3.imshow(umask, cmap='gray')
    image = drawContourOnImage(image, umask)
    ax4.imshow(image)
    ax1.axis('off')
    ax2.axis('off')
    ax3.axis('off')
    ax4.axis('off')
fig.subplots_adjust(hspace=0)
plt.subplot_tool()
plt.show()