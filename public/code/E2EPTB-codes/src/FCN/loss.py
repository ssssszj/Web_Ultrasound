import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class BCELoss():
    def __init__(self):
        self.bce = nn.BCELoss()

        if torch.cuda.is_available():
            self.bce = self.bce.cuda()

    def dice_loss(self, pred, target, smooth = 1.):
        iflat = pred.contiguous().view(-1)
        tflat = target.contiguous().view(-1)
        intersection = (iflat * tflat).sum()
        
        loss = 1 - ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))
        
        return loss.mean()

    def loss(self, seg_preds, seg_targets, lam_seg=0.2):
        isGPU = torch.cuda.is_available()
        seg_loss = self.bce(seg_preds.to(torch.double), seg_targets.to(torch.double))
        dice_loss = self.dice_loss(seg_preds.to(torch.double), seg_targets.to(torch.double))
        
        loss = lam_seg * seg_loss + (1-lam_seg) * dice_loss

        return loss, seg_loss

    def weighted_binary_cross_entropy(self, output, target):    
        if self.weights is not None:
            assert len(self.weights) == 2
            
            loss = self.weights[1] * (target * torch.log(output)) + \
                self.weights[0] * ((1 - target) * torch.log(1 - output))
        else:
            loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)

        return torch.neg(torch.mean(loss))