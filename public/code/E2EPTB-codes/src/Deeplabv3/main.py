import torch.optim as optim
from sklearn.metrics import roc_auc_score, f1_score
from model import createDeepLabv3
from trainer import train_model
from utils import jaccard
import numpy as np
import datahandler
import argparse
import os
import torch

def dice_loss(pred, target, smooth = 1.):
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()
    
    loss = 1 - ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))
    
    return loss.mean()

def loss(seg_preds, seg_targets, lam_seg=0.2):
    bce = torch.nn.BCELoss()
    seg_loss = bce(seg_preds.to(torch.double), seg_targets.to(torch.double))
    
    dloss = dice_loss(seg_preds.to(torch.double), seg_targets.to(torch.double))
    loss = lam_seg * seg_loss + (1-lam_seg) * dloss

    return loss
    

"""
    Version requirements:
        PyTorch Version:  1.2.0
        Torchvision Version:  0.4.0a0+6b959ee
"""

# Command line arguments 
parser = argparse.ArgumentParser()
parser.add_argument(
    "data_directory", help='Specify the dataset directory path')
parser.add_argument(
    "exp_directory", help='Specify the experiment directory where metrics and model weights shall be stored.')
parser.add_argument("--epochs", default=500, type=int)
parser.add_argument("--batchsize", default=4, type=int)

args = parser.parse_args()


bpath = args.exp_directory
data_dir = args.data_directory
epochs = args.epochs
batchsize = args.batchsize
# Create the deeplabv3 resnet101 model which is pretrained on a subset of COCO train2017, on the 20 categories that are present in the Pascal VOC dataset.
model = createDeepLabv3()
model.train()
# Create the experiment directory if not present
if not os.path.isdir(bpath):
    os.mkdir(bpath)


# Specify the loss function
criterion = torch.nn.MSELoss(reduction='mean')
# Specify the optimizer with a lower learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Specify the evalutation metrics
metrics = {'iou': jaccard}


# Create the dataloader
dataloaders = datahandler.get_dataloader_single_folder(
    data_dir, imageFolder="Resized_img", maskFolder="Resized_masks", batch_size=batchsize)
trained_model = train_model(model, loss, dataloaders,
                            optimizer, bpath=bpath, metrics=metrics, num_epochs=epochs)


# Save the trained model
# torch.save({'model_state_dict':trained_model.state_dict()},os.path.join(bpath,'weights'))
torch.save(model, os.path.join(bpath, 'weights.pt'))
