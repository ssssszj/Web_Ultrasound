import sys
# sys.path.append('../src')

import numpy as np
import cv2
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from data_loader import CervixDataset
import datahandler
from PIL import Image
from sklearn.model_selection import train_test_split
from utils import get_samples_for_label, jaccard

cervix_dataset = CervixDataset("../../data/Bezier", "../../data/Beziermask", "../../data/annotations_final.csv")
dl = torch.load('./deeplab.pt', map_location="cpu")
    
def extract_cervix(image, model):
    orginal_shape = image.size
    image.show()
    image = image.resize((512, 512))
    inputs = Variable(transforms.ToTensor()(image).unsqueeze(0))
    pred_mask = model(inputs)
    print(pred_mask)
    pred_mask = torch.sigmoid(pred_mask['out'])
    outputs = pred_mask.squeeze(0).cpu().data
    mask = transforms.ToPILImage()(outputs)
    background = Image.new('RGB', (512, 512), color='black')

    return mask.resize(orginal_shape), image.resize(orginal_shape) #Image.composite(image, background, mask).resize(orginal_shape)

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

train_indices, test_indices = train_test_split(np.arange(len(cervix_dataset)), test_size=0.2, random_state=42)

images = [cervix_dataset[i][0] for i in train_indices[8:12]]
masks = [cervix_dataset[i][1] for i in train_indices[8:12]]

fig, axis = plt.subplots(len(images), 4, figsize=(10, 10), sharex=True, sharey=True)
axis[0][0].set_title("Input")
axis[0][1].set_title("GD")
axis[0][2].set_title("Mask")
axis[0][3].set_title("Result")
for image, mask, (ax1, ax2, ax3, ax4) in zip(images, masks, axis):
    umask, _ = extract_cervix(image, dl)
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
    # print(jaccard(np.array(umask), np.array(_mask)))
fig.subplots_adjust(hspace=0)
plt.subplot_tool()
plt.show()