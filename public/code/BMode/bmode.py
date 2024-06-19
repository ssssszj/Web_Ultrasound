import argparse
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.transforms import transforms
from torchvision.utils import make_grid
from src.dataset import *
from models import *
from src.utils import progress_bar


parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

print('==> Preparing data..')
train_dataset = MyDataset(healthy_dir="/mnt/data/data1/masks/train_healthy/",
                          sick_dir="/mnt/data/data1/masks/train_zaochan/",
                          transform=transforms.Compose(
                              [Resize((256, 256)),
                               ToGray(),
                               RandomCrop(),
                               RandomHorizontalFlip(),
                               ToTensor()]
                          ))
trainloader = DataLoader(dataset=train_dataset,
                                 batch_size=4,
                                 shuffle=True,
                                 num_workers=1)
test_dataset = MyDataset(healthy_dir="/mnt/data/data1/masks/test_healthy/",
                          sick_dir="/mnt/data/data1/masks/test_zaochan/",
                          transform=transforms.Compose(
                              [Resize((256, 256)),
                               ToGray(),
                               ToTensor()],
                          ))
testloader = DataLoader(dataset=test_dataset,
                                 batch_size=4,
                                 shuffle=False,
                                 num_workers=1)

classes = ('healthy','sick')

# Model
print('==> Building model..')
# net = LeNet()
# net = AlexNet()
# net = VGG('VGG11')
# net = GoogLeNet()
net = ResNet18()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']


# criterion = nn.CrossEntropyLoss()
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([275./132.]).to(device))
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    total_healthy = 0
    total_sick = 0
    correct_healthy = 0
    correct_sick = 0

    for batch_idx, sample in enumerate(trainloader):
       
        inputs = sample["image"]
        targets = sample["label"].float()
        targets =targets.unsqueeze(1)
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        predicted = torch.where(torch.sigmoid(outputs)>=0.5,1.,0.)
        targets = targets.squeeze()
        predicted = predicted.squeeze()
        total += targets.size(0)
        total_sick += targets.sum().item()
        total_healthy += targets.size(0) - targets.sum().item()
        correct += predicted.eq(targets).sum().item()
        healthy_mask = torch.where(targets == 0)
        sick_mask = torch.where(targets == 1)
        correct_healthy += predicted[healthy_mask].eq(targets[healthy_mask]).sum().item()
        correct_sick += predicted[sick_mask].eq(targets[sick_mask]).sum().item()
        acc_healthy = -1
        if total_healthy > 0:
            acc_healthy = 100.* correct_healthy/ total_healthy
        acc_sick = -1
        if total_sick > 0:
            acc_sick = 100.*correct_sick/total_sick

        progress_bar(batch_idx, len(trainloader),
                     'Loss: %.3f | Acc: %.3f%% (%d/%d) '
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    print(f'Acc_healthy: {acc_healthy} ({correct_healthy}/{total_healthy})')
    print(f'Acc_sick: {acc_sick} ({correct_sick}/{total_sick})')

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    total_healthy = 0
    total_sick = 0
    correct_healthy = 0
    correct_sick = 0
    with torch.no_grad():
        for batch_idx, sample in enumerate(testloader):
            inputs = sample["image"]
            targets = sample["label"].float()
            targets = targets.unsqueeze(1)
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            predicted = torch.where(torch.sigmoid(outputs)>=0.5,1.,0.)

            targets = targets.squeeze()
            predicted = predicted.squeeze()
            total += targets.size(0)
            total_sick += targets.sum().item()
            total_healthy += targets.size(0) - targets.sum().item()
            correct += predicted.eq(targets).sum().item()
            healthy_mask = torch.where(targets == 0)
            sick_mask = torch.where(targets == 1)
            correct_healthy += predicted[healthy_mask].eq(targets[healthy_mask]).sum().item()
            correct_sick += predicted[sick_mask].eq(targets[sick_mask]).sum().item()
            acc_healthy = -1
            if total_healthy > 0:
                acc_healthy = 100. * correct_healthy / total_healthy
            acc_sick = -1
            if total_sick > 0:
                acc_sick = 100. * correct_sick / total_sick

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        print(f'Acc_healthy: {acc_healthy} ({correct_healthy}/{total_healthy})')
        print(f'Acc_sick: {acc_sick} ({correct_sick}/{total_sick})')

    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc

for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    test(epoch)
    scheduler.step()