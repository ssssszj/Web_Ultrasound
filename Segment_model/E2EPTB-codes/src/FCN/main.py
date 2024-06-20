from __future__ import print_function

import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch
import torch.nn as nn
import torch.optim as optim
from loss import BCELoss
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from data_loader import CervixDataset, ImbalancedDatasetSampler
from utils import jaccard, dice, splitDataset

from fcn import VGGNet, FCN32s, FCN16s, FCN8s, FCNs

from matplotlib import pyplot as plt
import datetime
import visdom
import numpy as np
import time
import sys
import os

best_iou   = 0
n_class    = 1
batch_size = 4
epochs     = 500
lr         = 1e-3
momentum   = 0
w_decay    = 1e-5
step_size  = 40
gamma      = 0.5
configs    = "FCNs-BCEWithLogits_batch{}_epoch{}_RMSprop_scheduler-step{}-gamma{}_lr{}_momentum{}_w_decay{}".format(batch_size, epochs, step_size, gamma, lr, momentum, w_decay)
print("Configs:", configs)

# create dir for model
model_dir = "models"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
model_path = os.path.join(model_dir, configs)

use_gpu = torch.cuda.is_available()
num_gpu = list(range(torch.cuda.device_count()))

train_transform = transforms.Compose([
    transforms.Resize(size=(512, 512)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.Resize(size=(512, 512)),
    transforms.ToTensor()
])

train_dataset = CervixDataset("../../data/Bezier", "../../data/Beziermask", "../../data/annotations_final.csv", train_transform)
val_dataset = CervixDataset("../../data/Bezier", "../../data/Beziermask", "../../data/annotations_final.csv", test_transform)
test_dataset = CervixDataset("../../data/Bezier", "../../data/Beziermask", "../../data/annotations_final.csv", test_transform)

train_indices, val_indices, test_indices = splitDataset(np.arange(len(train_dataset)))

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    sampler=ImbalancedDatasetSampler(train_dataset, indices=train_indices),
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    sampler=SubsetRandomSampler(val_indices),
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    sampler=SubsetRandomSampler(test_indices),
    num_workers=0
)

vgg_model = VGGNet(requires_grad=True, remove_fc=True)
fcn_model = FCNs(pretrained_net=vgg_model, n_class=n_class)

if use_gpu:
    ts = time.time()
    vgg_model = vgg_model.cuda()
    fcn_model = fcn_model.cuda()
    # fcn_model = nn.DataParallel(fcn_model, device_ids=num_gpu)
    print("Finish cuda loading, time elapsed {:.4f}".format(time.time() - ts))

criterion = BCELoss()
optimizer = optim.SGD(fcn_model.parameters(), lr=lr, momentum=momentum, weight_decay=w_decay)
scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)  # decay LR by a factor of 0.5 every 30 epochs

# create dir for score
score_dir = os.path.join("scores", configs)
if not os.path.exists(score_dir):
    os.makedirs(score_dir)
IU_scores    = np.zeros((epochs, n_class))
pixel_scores = np.zeros(epochs)

def train():
    for epoch in range(epochs):
        scheduler.step()
        total_ious = []
        total_loss = []

        print("============ Epoch {}/{} ==============".format(epoch, epochs))

        ts = time.time()
        for iter, (input, mask, _) in enumerate(train_loader):
            optimizer.zero_grad()

            if use_gpu:
                input = input.cuda()
                mask = mask.cuda()

            output = fcn_model(input)
            loss, seg_loss = criterion.loss(output, mask)
            loss.backward()
            optimizer.step()
            total_loss.append(loss.item())
            
            output = output.round()
            output_np = output.data.cpu().numpy()
            mask_np = mask.data.cpu().numpy()
            
            iou = jaccard(output, mask)
            _dice = dice(output, mask)
            total_ious.append(iou)

            if iter % step_size == 0:
                print("iter {}/{}, loss: {:.4f}, iou: {:.4f}, dice: {:.4f}".format(iter, len(train_loader), loss.item(), iou, _dice))
                
                if iter % 5*step_size == 0:
                # output_np = np.argmax(output_np, axis=1)
                    writer.add_images("train_imgs_" + str(epoch) + "_" + str(iter) + '_iou:' + str(iou), 
                                      output_np, global_step=epoch+1, dataformats='NCHW')
        
        mloss = np.array(total_loss).mean()
        mios = np.array(total_ious).mean()
        print("train: epoch {}, loss: {:.4f}, iou: {:.4f}".format(epoch, mloss, mios))
        writer.add_scalars('iou', {'train': mios}, global_step=epoch+1)
        writer.add_scalars('loss', {'train': mloss}, global_step=epoch+1)
        
        print("Finish epoch {}, time elapsed {:.4f}".format(epoch, time.time() - ts))

        val(epoch)
        test(epoch)

def val(epoch):
    fcn_model.eval()
    total_ious = []
    total_loss = []
    for iter, (input, mask, _) in enumerate(val_loader):
        if use_gpu:
            input = input.cuda()
            mask = mask.cuda()

        output = fcn_model(input)
        loss, seg_loss = criterion.loss(output, mask)
        output = output.round()
        output_np = output.data.cpu().numpy()
        mask_np = mask.data.cpu().numpy()
        
        total_loss.append(loss.item())
        iou = jaccard(output, mask)
        total_ious.append(iou)
    
    mloss = np.array(total_loss).mean()
    mios = np.array(total_ious).mean()
    print("val: epoch {}, loss: {:.4f}, iou: {:.4f}".format(epoch, mloss, mios))
    writer.add_scalars('iou', {'val': mios}, global_step=epoch+1)
    writer.add_scalars('loss', {'val': mloss}, global_step=epoch+1)
    
    global best_iou
    if mios > best_iou:
        best_iou = mios
        torch.save(fcn_model, model_path)
    
def test(epoch):
    fcn_model.eval()
    total_ious = []
    total_loss = []
    for iter, (input, mask, _) in enumerate(test_loader):
        if use_gpu:
            input = input.cuda()
            mask = mask.cuda()

        output = fcn_model(input)
        loss, seg_loss = criterion.loss(output, mask)
        output = output.round()
        output_np = output.data.cpu().numpy()
        mask_np = mask.data.cpu().numpy()

        total_loss.append(loss.item())
        iou = jaccard(output, mask)
        total_ious.append(iou)
    
    mloss = np.array(total_loss).mean()
    mios = np.array(total_ious).mean()
    print("test: epoch {}, loss: {:.4f}, iou: {:.4f}".format(epoch, mloss, mios))
    writer.add_scalars('iou', {'test': mios}, global_step=epoch+1)
    writer.add_scalars('loss', {'test': mloss}, global_step=epoch+1)


if __name__ == "__main__":
    now = datetime.datetime.now()
    strtime = now.strftime("%d-%m-%Y_%H:%M:%S")
    writer = SummaryWriter("logs/logs_" + strtime)
    
    # vis = visdom.Visdom()
    # val(0)  # show the accuracy before training
    train()
