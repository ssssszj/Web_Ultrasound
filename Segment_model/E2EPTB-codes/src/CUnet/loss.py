import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def CrossEntropyLoss2d():
    def wrap(seg_preds, seg_targets, class_inputs, class_targets,
             lambda_1=1.0, lambda_2=0.1, pixel_average=True):

        n, c, h, w = seg_preds.size()

        # Calculate segmentation loss
        seg_inputs = seg_preds.transpose(1, 2).transpose(2, 3).contiguous()
        seg_inputs = seg_inputs[seg_targets.view(n, h, w, 1).repeat(1, 1, 1, c) > 0].view(-1, c)

        # Exclude the 0-valued pixels from the loss calculation as 0 values represent the pixels that are not annotated.
        seg_targets_mask = seg_targets > 0
        # Subtract 1 from all classes, in the ground truth tensor, in order to match the network predictions.
        # Remember, in network predictions, label 0 corresponds to label 1 in ground truth.
        seg_targets = seg_targets[seg_targets_mask] - 1

        # Calculate segmentation loss value using cross entropy
        seg_loss = F.cross_entropy(seg_inputs, seg_targets, size_average=False)

        # Average the calculated loss value over each labeled pixel in the ground-truth tensor
        if pixel_average:
            seg_loss /= seg_targets_mask.float().data.sum()
        loss = lambda_1 * seg_loss

        # Calculate class loss, multiply with coefficient, lambda_2, sum with total loss
        
        # Calculate classification loss
        class_targets -= 1
        class_loss = F.cross_entropy(class_inputs, class_targets)
        # Combine losses
        loss += lambda_2 * class_loss

        seg_loss = seg_loss.item()
        class_loss = class_loss.item()

        return loss, seg_loss, class_loss
    return wrap

class BCELoss():
    def __init__(self, args, weights):
        self.weights = weights
        self.bce = nn.BCELoss()

        if args.onGPU:
            self.bce = self.bce.cuda()

    def dice_loss(self, pred, target, smooth = 1.):
        pred = pred.contiguous()
        target = target.contiguous()    

        intersection = (pred * target).sum(dim=2).sum(dim=2)
        loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
        
        return loss.mean()

    def loss(self, seg_preds, seg_targets, class_input, class_target, lam_seg=0.8, lam_class=0.1):
        # seg_loss = self.weighted_binary_cross_entropy(seg_preds, seg_targets)
        # print(class_input, class_target)
        # class_loss = self.weighted_binary_cross_entropy(class_input, class_target)

        isGPU = torch.cuda.is_available()

        seg_loss = self.bce(seg_preds.to(torch.double), seg_targets.to(torch.double))

        # weights_ = Variable(self.weights[class_target.data.view(-1).long()].view_as(class_target))
        # if isGPU:
        #     weights_ = weights_.cuda()

        # class_loss = self.bce(class_input.to(torch.double), class_target.to(torch.double)) #* weights_
        class_loss = self.weighted_binary_cross_entropy(class_input.to(torch.double), class_target.to(torch.double))
        dice_loss = self.dice_loss(seg_preds.to(torch.double), seg_targets.to(torch.double))
        loss = lam_seg * seg_loss + (1-lam_seg) * dice_loss + lam_class * class_loss

        return loss, seg_loss, class_loss

    def weighted_binary_cross_entropy(self, output, target):    
        if self.weights is not None:
            assert len(self.weights) == 2
            
            loss = self.weights[1] * (target * torch.log(output)) + \
                self.weights[0] * ((1 - target) * torch.log(1 - output))
        else:
            loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)

        return torch.neg(torch.mean(loss))