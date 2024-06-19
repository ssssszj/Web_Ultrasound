import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import sys
sys.path.append('../src/UNet/src/PyTorch')

import time

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from sklearn.model_selection import train_test_split
from data_loader import CervixDataset
from tensorboardX import SummaryWriter

from models import UNet

import argparse



writer = SummaryWriter('logs')

train_transform = transforms.Compose([
    transforms.Resize(size=(256, 256)),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.Resize(size=(256, 256)),
    transforms.ToTensor()
])


parser = argparse.ArgumentParser(description='UNet Training Script')
parser.add_argument('--data_path', type=str, default='../data/Bezier', help='Path to the dataset folder')
parser.add_argument('--mask_path', type=str, default='../data/Beziermask', help='Path to the mask folder')
parser.add_argument('--annotations_file', type=str, default='../data/annotations_final.csv', help='Path to the annotations CSV file')
args = parser.parse_args()

train_dataset = CervixDataset(args.data_path, args.mask_path, args.annotations_file, train_transform)
val_dataset = CervixDataset(args.data_path, args.mask_path, args.annotations_file, test_transform)

#train_dataset = CervixDataset("../data/Bezier", "../data/Beziermask", "../data/annotations_final.csv", train_transform)
#val_dataset = CervixDataset("../data/Bezier", "../data/Beziermask", "../data/annotations_final.csv", test_transform)
train_indices, test_indices = train_test_split(np.arange(len(train_dataset)), test_size=0.2, random_state=42)

train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    sampler=SubsetRandomSampler(train_indices),
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=4,
    sampler=SubsetRandomSampler(test_indices),
    num_workers=0
)

print(len(train_loader))

def jaccard(outputs, targets):
    outputs = outputs.view(outputs.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    intersection = (outputs * targets).sum(1)
    union = (outputs + targets).sum(1) - intersection
    jac = (intersection + 0.001) / (union + 0.001)
    return jac.mean()

# add this line for Mac without CUDA support
cuda = True if torch.cuda.is_available() else False

model = UNet()
model = model.cuda() if cuda else model


criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

model_folder = os.path.abspath('../models')
if not os.path.exists(model_folder):
    os.mkdir(model_folder)
model_path = os.path.join(model_folder, 'unet.pt')

hist = {'loss': [], 'jaccard': [], 'val_loss': [], 'val_jaccard': []}
num_epochs = 120
display_steps = 2
best_jaccard = 0



#==============================================================#
#                           TRAIN                              #
#==============================================================#

# Start time of learning
total_start_training = time.time()

for epoch in range(num_epochs):
    start_time_epoch = time.time()
    print('Starting epoch {}/{}'.format(epoch + 1, num_epochs))
    # train
    model.train()
    running_loss = 0.0
    running_jaccard = 0.0
    for batch_idx, (images, masks, _) in enumerate(train_loader):
        images = Variable(images.cuda() if cuda else images)
        masks = Variable(masks.cuda() if cuda else masks)

        optimizer.zero_grad()
        outputs, _ = model(images)
        predicted = outputs.round()
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        jac = jaccard(outputs.round(), masks)
        running_jaccard += jac.item()
        running_loss += loss.item()

        if batch_idx % display_steps == 0:
            print('    ', end='')
            print('batch {:>3}/{:>3} loss: {:.4f}, jaccard {:.4f}, learning time:  {:.2f}s\r' \
                  .format(batch_idx + 1, len(train_loader),
                          loss.item(), jac.item(), time.time() - start_time_epoch))

    # evalute
    print('Finished epoch {}, starting evaluation'.format(epoch + 1))
    model.eval()
    val_running_loss = 0.0
    val_running_jaccard = 0.0
    for images, masks, _ in val_loader:
        images = Variable(images.cuda() if cuda else images)
        masks = Variable(masks.cuda() if cuda else masks)

        outputs, _ = model(images)
        loss = criterion(outputs, masks)

        val_running_loss += loss.item()
        jac = jaccard(outputs.round(), masks)
        val_running_jaccard += jac.item()


    train_loss = running_loss / len(train_loader)
    val_loss = val_running_loss / len(val_loader)
    writer.add_scalars('loss', {'train': train_loss, 'val': val_loss}, global_step=epoch+1)

    train_jaccard = running_jaccard / len(train_loader)
    val_jaccard = val_running_jaccard / len(val_loader)
    writer.add_scalars('jacc', {'train': train_jaccard, 'val': val_jaccard}, global_step=epoch+1)

    hist['loss'].append(train_loss)
    hist['jaccard'].append(train_jaccard)
    hist['val_loss'].append(val_loss)
    hist['val_jaccard'].append(val_jaccard)

    if val_jaccard > best_jaccard:
        torch.save(model, model_path)
    print('    ', end='')
    print('loss: {:.4f}  jaccard: {:.4f} \
           val_loss: {:.4f} val_jaccard: {:4.4f}\n' \
          .format(train_loss, train_jaccard, val_loss, val_jaccard))

print('Training UNet finished, took {:.2f}s'.format(time.time() - total_start_training))