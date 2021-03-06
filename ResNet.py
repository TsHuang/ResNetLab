import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn.init as init
import numpy as np

from torch.autograd import Variable


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        #init.kaiming_normal(self.conv1.weight)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        #init.kaiming_normal(self.conv2.weight)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class BasicBlockVanilla(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlockVanilla, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        #init.kaiming_normal(self.conv1.weight)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        #init.kaiming_normal(self.conv2.weight)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out)
        return out

class BasicBlockPreActive(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlockPreActive, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)
        return out

class ResNetTs(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNetTs, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64*block.expansion, num_classes)
        self.reset_params()
        #self.linear = nn.Linear(64, num_classes) # not sure about this line
        #print(block.expansion)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1) # stride only done in the first block
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion # why?
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, mode = 'fan_out')
                if m.bias is not None:
                    init.constant(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    init.constant(m.weight, 1)
                    init.constant(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    init.normal(m.weight, std=0.001)
                    if m.bias is not None:
                        init.constant(m.bias, 0)

class VanillaNetTs(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(VanillaNetTs, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        #self.linear = nn.Linear(64*block.expansion, num_classes)
        self.linear = nn.Linear(64, num_classes) # not sure about this line
        #print(block.expansion)
        self.reset_params()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1) # stride only done in the first block
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion # why?
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    init.constant(m.weight, 1)
                    init.constant(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    init.normal(m.weight, std=0.001)
                    if m.bias is not None:
                        init.constant(m.bias, 0)

class ResNetPreActiveTs(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNetPreActiveTs, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.bn1 = nn.BatchNorm2d(64*block.expansion)
        self.linear = nn.Linear(64*block.expansion, num_classes)
        self.reset_params()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1) # stride only done in the first block
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion # why?
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, mode = 'fan_out')
                if m.bias is not None:
                    init.constant(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    init.constant(m.weight, 1)
                    init.constant(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    init.normal(m.weight, std=0.001)
                    if m.bias is not None:
                        init.constant(m.bias, 0)


def ResNet20():
    return ResNetTs(BasicBlock, [3,3,3])

def ResNet56():
    return ResNetTs(BasicBlock, [9,9,9])

def ResNet110():
    return ResNetTs(BasicBlock, [16,16,16])

def VanillaNet20():
    return VanillaNetTs(BasicBlockVanilla, [3,3,3])

def VanillaNet56():
    return VanillaNetTs(BasicBlockVanilla, [9,9,9])

def VanillaNet110():
    return VanillaNetTs(BasicBlockVanilla, [16,16,16])

def ResNetPreActive20():
    return ResNetPreActiveTs(BasicBlockPreActive, [3,3,3])

def ResNetPreActive56():
    return ResNetPreActiveTs(BasicBlockPreActive, [9,9,9])

def ResNetPreActive110():
    return ResNetPreActiveTs(BasicBlockPreActive, [16,16,16])

def test():
    #net = ResNet20()
    #net = VanillaNet20()
    net = ResNetPreActive20()

    #y = net()
    print(net)
    params = list(net.parameters())
    print(len(params))
    print(params[0].size())

    input = Variable(torch.randn(1, 3, 32, 32))
    out = net(input)
    print(out)


test()