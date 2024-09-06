import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class BottleneckBlock(nn.Module):
    expansion = 4  # The output channels are 4 times the input channels
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(BottleneckBlock, self).__init__()
        #1. Convolutional Layers and Batch Normalization
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=stride,padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels,out_channels,padding=1, kernel_size=3, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(out_channels,out_channels*4,kernel_size=1,stride=stride,padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        #2. Downsampling Sample
        self.downsample = None
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )
        
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class ResnetModel(nn.Module):
    def __init__(self) -> None:
        super(ResnetModel,self).__init__()
        self.in_chanels = 64
        
        self.conv1 = nn.Conv2d(in_channels=64,out_channels=64, kernel_size=7, stride=2, padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.layer1 = self._make_layer(64,3)
        self.layer2 = self._make_layer(128,4,strides=2)
        self.layer3 = self._make_layer(256,23,strides=2)
        self.layer4 = self._make_layer(512,3,strides=2)
        
        self.avrpool = nn.AdaptiveAvgPool2d((1,1))
        #Fully connected layer
        self.fc = nn.Linear(512 * BottleneckBlock.expansion, 1)
        self.sigmoid = nn.Sigmoid()
        
    def _make_layer(self,out_channels, block, strides = 1):
        layers = []
        layers.append(BottleneckBlock(self.in_chanels,out_channels,stride=strides))
        self.in_chanels = out_channels * BottleneckBlock.expansion
        for _ in range(1,block):
            layers.append(BottleneckBlock(self.in_chanels, out_channels))
        return nn.Sequential(*layers)
    
    def forwrad(self,x):
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        
        out = self.maxpool(out)
        out = self.layer1(out)
        
        out = self.layer2(out)
        
        out = self.layer3(out)
        
        out = self.layer4(out)
        
        out = self.avrpool(out)
        out = torch.flatten(out,1)
        out = self.fc(out)
        out = self.sigmoid(out)
        
        return out
        
        
        
        