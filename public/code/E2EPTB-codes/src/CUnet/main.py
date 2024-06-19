import os
import time
import torch
import datetime
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from argparse import ArgumentParser
from loss import BCELoss
from model import CUnet
from models import UNet
from AttentionModels import AttU_Net
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from data_loader import CervixDataset, ImbalancedDatasetSampler
from tensorboardX import SummaryWriter
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import train_test_split
from utils import jaccard, one_hot_vector, metrics, conf_matrix, auc, splitDataset

def train(args, train_loader, model, criterion, optimizer, epoch):
    model.train()

    running_loss = 0.0
    running_seg_loss = 0.0
    running_class_loss = 0.0
    running_jaccard = 0.0
    running_class_acc = 0.0
    running_class_recall = 0.0
    running_class_precision = 0.0
    y_true = list()
    y_pred = list()

    start_time_epoch = time.time()
    for i, (input, mask, label) in enumerate(train_loader):

        if args.onGPU:
            input = input.cuda()
            mask = mask.cuda()
            label = label.cuda()

        input = Variable(input)
        mask = Variable(mask)
        label = Variable(label)

        optimizer.zero_grad()
        output_mask, output_label = model(input)

        label_one_hot_vector = one_hot_vector(label, args.classes)
        if args.onGPU:
            label_one_hot_vector = label_one_hot_vector.cuda()

        loss, seg_loss, class_loss = criterion.loss(output_mask, mask, output_label, label_one_hot_vector, lam_seg=args.lambda_seg, lam_class=args.lambda_class)
        loss.backward()
        optimizer.step()

        jac = jaccard(output_mask.round(), mask)
        running_jaccard += jac.item()
        running_loss += loss.item()
        running_seg_loss += seg_loss.item()
        running_class_loss += class_loss.item()
        _, output_label = torch.max(output_label, 1)
        class_acc = np.mean((output_label == label).data.cpu().numpy())
        running_class_acc += class_acc

        _, precision, recall, f1 = metrics(label.cpu(), output_label.cpu())
        running_class_recall += recall
        running_class_precision += precision

        y_true.extend(label.cpu().numpy().flatten().tolist())
        y_pred.extend(output_label.cpu().numpy().flatten().tolist())

        if i % args.print_freq == 0:
            # print('    ', end='')
            print('Batch {:>3}/{:>3} loss: {:.4f}, learning time: {:.2f}s \
                \n\tSegmentation: seg_loss: {:.4f}, jaccard: {:.4f}, \
                \n\tClasification: class_loss: {:.4f}, class_acc: {:.4f}, class_recall: {:.4f}, class_precision: {:.4f} \r' \
                  .format(i + 1, len(train_loader), loss.item(), time.time() - start_time_epoch, seg_loss.item(), jac.item(), class_loss.item(), class_acc, recall, precision))

    return running_seg_loss, running_class_loss, running_loss, running_jaccard, running_class_acc, running_class_recall, running_class_precision, y_true, y_pred

        
def val(args, val_loader, model, criterion, epoch):
    model.eval()

    val_running_loss = 0.0
    val_running_seg_loss = 0.0
    val_running_class_loss = 0.0
    val_running_jaccard = 0.0
    val_running_class_acc = 0.0
    val_running_class_recall = 0.0
    val_running_class_precision = 0.0
    y_true = list()
    y_pred = list()

    for i, (input, mask, label) in enumerate(val_loader):
        start_time_epoch = time.time()

        if args.onGPU:
            input = input.cuda()
            mask = mask.cuda()
            label = label.cuda()

        input = Variable(input)
        mask = Variable(mask)
        label = Variable(label)

        output_mask, output_label = model(input)
        label_one_hot_vector = one_hot_vector(label, args.classes)
        if args.onGPU:
            label_one_hot_vector = label_one_hot_vector.cuda()
        loss, seg_loss, class_loss = criterion.loss(output_mask, mask, output_label, label_one_hot_vector, lam_seg=args.lambda_seg, lam_class=args.lambda_class)

        jac = jaccard(output_mask.round(), mask)
        val_running_jaccard += jac.item()
        val_running_loss += loss.item()
        val_running_seg_loss += seg_loss.item()
        val_running_class_loss += class_loss.item()
        _, output_label = torch.max(output_label, 1)
        val_running_class_acc += np.mean((output_label == label).data.cpu().numpy())

        _, precision, recall, f1 = metrics(label.cpu(), output_label.cpu())
        val_running_class_recall += recall
        val_running_class_precision += precision

        y_true.extend(label.cpu().numpy().flatten().tolist())
        y_pred.extend(output_label.cpu().numpy().flatten().tolist())

    return val_running_seg_loss, val_running_class_loss, val_running_loss, val_running_jaccard, val_running_class_acc, val_running_class_recall, val_running_class_precision, y_true, y_pred

def test(args, test_loader, model, criterion):
    model.eval()

    test_running_loss = 0.0
    test_running_seg_loss = 0.0
    test_running_class_loss = 0.0
    test_running_jaccard = 0.0
    y_true = list()
    y_pred = list()

    for i, (input, mask, label) in enumerate(test_loader):
        start_time_epoch = time.time()

        if args.onGPU:
            input = input.cuda()
            mask = mask.cuda()
            label = label.cuda()

        input = Variable(input)
        mask = Variable(mask)
        label = Variable(label)

        output_mask, output_label = model(input)
        label_one_hot_vector = one_hot_vector(label, args.classes)
        if args.onGPU:
            label_one_hot_vector = label_one_hot_vector.cuda()
        loss, seg_loss, class_loss = criterion.loss(output_mask, mask, output_label, label_one_hot_vector, lam_seg=args.lambda_seg, lam_class=args.lambda_class)

        jac = jaccard(output_mask.round(), mask)
        test_running_jaccard += jac.item()
        test_running_loss += loss.item()
        test_running_seg_loss += seg_loss.item()
        test_running_class_loss += class_loss.item()
        _, output_label = torch.max(output_label, 1)

        y_true.extend(label.cpu().numpy().flatten().tolist())
        y_pred.extend(output_label.cpu().numpy().flatten().tolist())

    return test_running_seg_loss, test_running_class_loss, test_running_loss, test_running_jaccard, y_true, y_pred


def trainValidateSegmentation(args):
    model = CUnet(number_of_class=args.classes)
    if args.onGPU == True:
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)
        model = model.cuda()

    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # if args.onGPU == True:
    #     cudnn.benchmark = True

    total_start_training = time.time()

    now = datetime.datetime.now()
    str = now.strftime("%d-%m-%Y_%H:%M:%S")
    writer = SummaryWriter("logs/logs_" + str)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_epochs, gamma=0.1)

    if not os.path.exists(args.savedir):
        os.mkdir(args.savedir)
    model_path = os.path.join(args.savedir, 'unet.pt')

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
        batch_size=args.batch_size,
        sampler=ImbalancedDatasetSampler(train_dataset, indices=train_indices),
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=SubsetRandomSampler(val_indices),
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        sampler=SubsetRandomSampler(test_indices),
        num_workers=0
    )

    weights = train_dataset.class_weights()
    weights = torch.from_numpy(weights)
    print(weights)
    if args.onGPU == True:
        weights = weights.cuda()

    criterion = BCELoss(args, weights)

    best_jaccard = 0
    for epoch in range(args.max_epochs):
        start_time_epoch = time.time()

        print('Starting epoch {}/{}'.format(epoch + 1, args.max_epochs))
        scheduler.step(epoch)
        
        running_seg_loss, running_class_loss, running_loss, running_jaccard, running_class_acc, running_class_recall, running_class_precision, y_true, y_pred, = train(args, train_loader, model, criterion, optimizer, epoch)
        val_running_seg_loss, val_running_class_loss, val_running_loss, val_running_jaccard, val_running_class_acc, val_running_class_recall, val_running_class_precision, val_y_true, val_y_pred = val(args, val_loader, model, criterion, epoch)
        test_running_seg_loss, test_running_class_loss, test_running_loss, test_running_jaccard, test_y_true, test_y_pred = test(args, test_loader, model, criterion)

        train_loss = running_loss / len(train_loader)
        val_loss = val_running_loss / len(val_loader)
        test_loss = test_running_loss / len(test_loader)
        writer.add_scalars('loss', {'train': train_loss, 'val': val_loss, 'test': test_loss}, global_step=epoch+1)

        train_jaccard = running_jaccard / len(train_loader)
        val_jaccard = val_running_jaccard / len(val_loader)
        test_jaccard = test_running_jaccard / len(test_loader)
        writer.add_scalars('jacc', {'train': train_jaccard, 'val': val_jaccard, 'test': test_jaccard}, global_step=epoch+1)

        _, train_precision, train_recall, _ = metrics(y_true, y_pred, average='weighted')
        train_auc = auc(y_true, y_pred, average='weighted')
        _, val_precision, val_recall, _ = metrics(val_y_true, val_y_pred, average='weighted')
        val_auc = auc(val_y_true, val_y_pred, average='weighted')
        _, test_precision, test_recall, _ = metrics(test_y_true, test_y_pred, average='weighted')
        test_auc = auc(test_y_true, test_y_pred, average='weighted')

        writer.add_scalars('class_recall', {'train': train_recall, 'val': val_recall, 'test': test_recall}, global_step=epoch+1)
        writer.add_scalars('class_precision', {'train': train_precision, 'val': val_precision, 'test': test_precision}, global_step=epoch+1)
        writer.add_scalars('class_auc', {'train': train_auc, 'val': val_auc, 'test': test_auc}, global_step=epoch+1)

        if val_jaccard > best_jaccard:
            torch.save(model, model_path)
        print('=' * 50)
        print('loss: {:.4f}  jaccard: {:.4f} recall: {:.4f} precision: {:.4f} \
           \nval_loss: {:.4f} val_jaccard: {:4.4f} recall: {:.4f} precision: {:.4f} auc: {:.4f} \
           \ntest_loss: {:.4f} test_jaccard: {:4.4f} recall: {:.4f} precision: {:.4f} auc: {:.4f}' \
          .format(train_loss, train_jaccard, train_recall, train_precision, val_loss, val_jaccard, val_recall, val_precision, val_auc, test_loss, test_jaccard, test_recall, test_precision, test_auc))
        print("Val:")
        conf_matrix(val_y_true, val_y_pred)
        print("Test:")
        conf_matrix(test_y_true, test_y_pred)

    print('Training CUNet finished, took {:.2f}s'.format(time.time() - total_start_training))
    

if __name__ == '__main__':

    parser = ArgumentParser(description='Training CUNet')
    parser.add_argument('--max_epochs', type=int, default=100, help='Max. number of epochs')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for processing the data')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--lr_decay_epochs', type=int, default=25, help='decay the learning rate after these many epochs')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='momentum factor for optimizer')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--savedir', default='./results', help='results directory')
    parser.add_argument('--classes', type=int, default=2, help='Number of classes in the dataset')
    parser.add_argument('--lambda_seg', type=float, default=0.5, help='lambda value for segmentation loss')
    parser.add_argument('--lambda_class', type=float, default=0.8, help='lambda value for classification loss')
    parser.add_argument('--print_freq', type=int, default=10, help='log training accuracy and loss every nth iteration')
    parser.add_argument('--logFile', default='trainValLog.txt', help="Log file")
    parser.add_argument('--onGPU', default=True, help='True if you want to train on GPU')

    args = parser.parse_args()

    if args.onGPU:
        args.onGPU = torch.cuda.is_available()

    print(args)
    trainValidateSegmentation(args)