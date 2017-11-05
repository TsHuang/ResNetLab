'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import numpy as np

import torchvision
import torchvision.transforms as transforms

import os
import argparse

import sys
from os import makedirs
from os.path import isfile, join, exists

# from models import *
from ResNet import *
from utils import progress_bar
from torch.autograd import Variable

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--checkpt', '-c', help="Path to save the checkpoints to")
#parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--model', '-m', default=0, type=int, help='0:ResNet, 1:VanillaNet, 2:Bonus')
parser.add_argument('--layers', '-l', default=20, type=int, help='support only 20, 56, 110 layers')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')

args = parser.parse_args()

use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
report = np.array(['FinalTestAcc', 'TrainLossCurve', 'TrainErrCurve', 'TestErrCurve'])

#create folders
checkpoint_dir = args.checkpt
best_dir = checkpoint_dir + "/best"
resume_dir = checkpoint_dir + "/resume"

if not exists(checkpoint_dir):
    print("checkpoint directories not found, creating directories...")
    makedirs(checkpoint_dir)

if not exists(best_dir):
    print('best_dir=', best_dir)
    makedirs(best_dir)

if not exists(resume_dir):
    makedirs(resume_dir)

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    transforms.Normalize((0.4914, 0.4824, 0.4467), (0.2470, 0.2435, 0.2616)), #somehow this modification does not as good as the original one
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    transforms.Normalize((0.4914, 0.4824, 0.4467), (0.2470, 0.2435, 0.2616)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    #assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    assert os.path.isdir(resume_dir), 'Error: no resume directory found!'
    resume_file = join(resume_dir, "/resume.t7")
    #checkpoint = torch.load('./checkpoint/ResNet20/ResNet_118.t7')
    checkpoint = torch.load(resume_file)
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
else:
    print('==> Building model..')
    if args.model == 0: #ResNet
        if args.layers == 20:
            net = ResNet20()
            print('net = ResNet20')
        elif args.layers == 56:
            net = ResNet56()
            print('net = ResNet56')
        elif args.layers == 110:
            net = ResNet110()
            print('net = ResNet110')
    elif args.model == 1: #VanillaNet
        if args.layers == 20:
            net = VanillaNet20()
            print('net = VanillaNet20')
        elif args.layers == 56:
            net = VanillaNet56()
            print('net = VanillaNet56')
        elif args.layers == 110:
            net = VanillaNet110()
            print('net = VanillaNet110')


if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)


# learning rate scheduling
def adjust_learning_rate(optimizer, epoch):
    if epoch < 81:
        lr = 0.1
    elif epoch < 122:
        lr = 0.01
    else:
        lr = 0.001

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)

    net.train()
    adjust_learning_rate(optimizer, epoch)

    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    acc = 100. * correct / total
    return train_loss, (100 - acc)


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # Save checkpoint.
    acc = 100. * correct / total

    print('Saving..')
    savefilename = checkpoint_dir + '/ResNet_' + str(epoch) + '.t7'
    state = {
        'net': net.module if use_cuda else net,
        'acc': acc,
        'epoch': epoch,
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, savefilename)

    print('Loss: %.3f, Accuracy: %.3f' % (test_loss / (batch_idx + 1), acc))
    if acc > best_acc:
        print('Saving..')
        #savefilename = './checkpoint/ResNet110/best/ResNet_' + str(epoch) + '.t7'
        savefilename = best_dir + '/ResNet_' + str(epoch) + '.t7'

        state = {
            'net': net.module if use_cuda else net,
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, savefilename)
        best_acc = acc

    print('Best Accuracy: %.3f' % best_acc)
    return (100 - acc)


# run code

for epoch in range(start_epoch, start_epoch + 164):
    # for epoch in range(start_epoch, start_epoch + 2):

    trainloss, trainErr = train(epoch)
    testErr = test(epoch)
    newdata = np.array([best_acc, trainloss, trainErr, testErr])
    report = np.vstack((report, newdata))
    # report = ['FinalTestErr', 'TrainLossCurve', 'TestErrCurve']

# save curve data
curve_file = checkpoint_dir + '/curveData.csv'
np.savetxt(curve_file, report, fmt="%s", delimiter=",")
#np.savetxt("./checkpoint/ResNet110/report.csv", report, fmt="%s", delimiter=",")
