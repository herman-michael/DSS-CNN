
from typing import Tuple

import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from torchmetrics import R2Score
import matplotlib.cm as cm
from scipy.stats import skewnorm
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import math
import cv2

import spconv.pytorch as spconv
from spconv.pytorch import SparseConvTensor, SparseModule, SparseSequential

from torch.utils.tensorboard import SummaryWriter
from data_loading_sa_aug import RegressionTaskData
from sklearn.linear_model import LinearRegression
from numpy import pi

# Define custom scheduler class
class CosineAnnealingWarmRestartsDecay(CosineAnnealingWarmRestarts):
    def __init__(self, optimizer, T_0, T_mult=1,
                    eta_min=0, last_epoch=-1, verbose=False, decay=1):
        super().__init__(optimizer, T_0, T_mult=T_mult,
                            eta_min=eta_min, last_epoch=last_epoch, verbose=verbose)
        self.decay = decay
        self.initial_lrs = self.base_lrs
    
    def step(self, epoch=None):
        if epoch == None:
            if self.T_cur + 1 == self.T_i:
                if self.verbose:
                    print("multiplying base_lrs by {:.4f}".format(self.decay))
                self.base_lrs = [base_lr * self.decay for base_lr in self.base_lrs]
        else:
            if epoch < 0:
                raise ValueError("Expected non-negative epoch, but got {}".format(epoch))
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    n = int(epoch / self.T_0)
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
            else:
                n = 0
            
            self.base_lrs = [initial_lrs * (self.decay**n) for initial_lrs in self.initial_lrs]

        super().step(epoch)

class DenseBasicRegression(nn.Module):
    """
    Basic dense CNN model for the regression task
    """
    def __init__(self, image_size: Tuple[int, int, int] = (1, 512, 512)):
        super(DenseBasicRegression, self).__init__()
        self.image_size = image_size
        self.conv1 = nn.Sequential(
        nn.Conv2d(in_channels=self.image_size[0], out_channels=4, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.regressor = nn.Sequential(
            nn.Linear(in_features=256,out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=2)
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = x.reshape(x.size(0), -1)
        x = self.regressor(x)
        return x
    
class DenseVGG13Regression(nn.Module):
    """
    This is a modified vgg13 model we will use for the regression task.
    """
    def __init__(self, image_size: Tuple[int, int, int] = (1, 512, 512)):
        super(DenseVGG13Regression, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(64),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer7 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer8 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer9 = nn.Sequential(
            nn.Conv2d(512, 4, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(4),
            nn.ReLU())
        self.layer10 = nn.Sequential(
            nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(4),
            nn.ReLU())

        self.regressor = nn.Sequential(
            nn.LazyLinear(out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=2)
        )
          
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = x.reshape(x.size(0), -1)
        x = self.regressor(x)
        return x
    
class DenseVGG16Regression(nn.Module):
    """
    This is a vgg16 model we will use for the regression task.
    """
    def __init__(self, image_size: Tuple[int, int, int] = (1, 512, 512)):
        super(DenseVGG16Regression, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(64),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer7 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer8 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer9 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer10 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer11 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer12 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer13 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        
        self.regressor = nn.Sequential(
            nn.Linear(in_features=256, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=2)
        )
          
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        x = self.layer12(x)
        x = self.layer13(x)
        x = x.reshape(x.size(0), -1)
        x = self.regressor(x)
        return x
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out
    
class ResNet(nn.Module):
    def __init__(self, block, layers):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
                        nn.Conv2d(1, 64, kernel_size = 7, stride = 2, padding = 3),
                        nn.BatchNorm2d(64),
                        nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride = 1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride = 2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride = 2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride = 2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        
        self.regressor = nn.Sequential(
            nn.LazyLinear(out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=2)
        )

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:

            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, class_input):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.regressor(x)

        return x

def replace_feature(out: SparseConvTensor,
                    new_features: SparseConvTensor) -> SparseConvTensor:
    if 'replace_feature' in out.__dir__():
        # spconv 2.x behaviour
        return out.replace_feature(new_features)
    else:
        out.features = new_features
        return out

class SparseBasicBlock(spconv.SparseModule):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, indice_key=None):
        super(SparseBasicBlock, self).__init__()
        algo=spconv.ConvAlgo.MaskImplicitGemm
        if stride == 1:
            self.conv1 = spconv.SubMConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, algo=algo, indice_key=indice_key).to(device)
        else:
            self.conv1 = spconv.SparseConv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False, algo=algo).to(device)
        #self.bn1 = nn.BatchNorm1d(out_channels).to(device)
        self.relu1 = nn.ReLU().to(device)
        self.conv2 = spconv.SubMConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, algo=algo, indice_key=indice_key).to(device)
        #self.bn2 = nn.BatchNorm1d(out_channels).to(device)
        self.relu2 = nn.ReLU().to(device)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        #out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu1(out.features))

        out = self.conv2(out)
        #out = replace_feature(out, self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = replace_feature(out, out.features + identity.features)
        out = replace_feature(out, self.relu2(out.features))

        return out

class SparseResNet34(nn.Module):
    def __init__(self, block, layers, num_classes=4, class_embedding_size=128):
        super(SparseResNet34, self).__init__()
        algo=spconv.ConvAlgo.MaskImplicitGemm
        self.in_channels = 64
        self.conv1 = spconv.SparseConv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False, indice_key="conv1").to(device)
        #self.bn1 = nn.BatchNorm1d(64).to(device)
        self.relu = nn.ReLU(inplace=True).to(device)
        self.maxpool = spconv.SparseMaxPool2d(kernel_size=3, stride=2, padding=1).to(device)
        self.layer1 = self._make_layer(block, 64, layers[0], indice_key="layer1")
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, indice_key="layer2")
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, indice_key="layer3")
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, indice_key="layer4")
        self.todense = spconv.SparseSequential(spconv.ToDense()).to(device)
        self.avgpool = nn.AvgPool2d(7, stride=1).to(device)
        self.regressor = nn.Sequential(
            nn.LazyLinear(out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=2),
            nn.Tanh()
        )
        self.class_embedding = nn.Sequential(
            nn.Linear(num_classes, class_embedding_size),
            nn.ReLU(),
            nn.Linear(class_embedding_size, class_embedding_size),
            nn.ReLU(),
        )

    def _make_layer(self, block, out_channels, blocks, stride=1, indice_key=None):
        downsample = None
        algo=spconv.ConvAlgo.MaskImplicitGemm
        if stride != 1 or self.in_channels != out_channels:
            downsample = spconv.SparseSequential(
                spconv.SparseConv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False, algo=algo, indice_key=indice_key).to(device),
                #nn.BatchNorm1d(out_channels),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample, indice_key=indice_key))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, indice_key=indice_key))

        return SparseSequential(*layers)

    def forward(self, x, class_input):
        x_sp = spconv.SparseConvTensor.from_dense(x.reshape(-1, 512, 512, 1))
        x = self.conv1(x_sp)
        #x = replace_feature(x, self.bn1(x.features))
        x = replace_feature(x, self.relu(x.features))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.todense(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        #  Convert to LongTensor and one-hot encode class inputs
        class_input = F.one_hot(class_input.long() - 1, num_classes=4).float()
        class_input = self.class_embedding(class_input)
        x = torch.cat((x, class_input), dim=1)
        x = self.regressor(x)
        return x
    

class SparseBasicRegression(nn.Module):
    """
    Basic sparse CNN model for the regression task
    """
    def __init__(self, image_size: Tuple[int, int, int] = (1, 512, 512)):
        super(SparseBasicRegression, self).__init__()
        self.image_size = image_size
        algo=spconv.ConvAlgo.MaskImplicitGemm
        self.conv1 = spconv.SparseSequential(
        spconv.SparseConv2d(in_channels=self.image_size[0], out_channels=4,
                            kernel_size=3, padding=1,
                            algo=algo).to(device),
        nn.ReLU(),
        spconv.SparseMaxPool2d(kernel_size=2, stride=2),
        spconv.SubMConv2d(in_channels=4, out_channels=16,
                          kernel_size=3, padding=1,
                          algo=algo,
                          indice_key="subm0").to(device),
        nn.ReLU(),
        spconv.SparseMaxPool2d(kernel_size=2, stride=2),
        spconv.ToDense(),
        )
        
        self.regressor = nn.Sequential(
            nn.LazyLinear(out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=2),
            nn.Tanh() 
        )

    def forward(self, x, class_input):
        # Create sparse tensor
        # you must make a batch axis before call to_sparse
        #torchTensorSp = x.to_sparse() # no channel axis here. equalivant to torchTensor.ndim
        #indices_th = torchTensorSp.indices().permute(1, 0).contiguous().int()
        # sparse tensor features need to have one channel axis.
        #features_th = torchTensorSp.values().view(-1, 1)
        # sparse tensor must have a batch axis, spatial shape dont contain batch axis.
        #x_sp = spconv.SparseConvTensor(features_th, indices_th, x.shape[1:], batch_size)
        x_sp = spconv.SparseConvTensor.from_dense(x.reshape(-1, 512, 512, 1))
        
        # Sparse convolutional layers
        x = self.conv1(x_sp)
        x = x.reshape(x.size(0), -1)
        x = self.regressor(x)
        return x
    
class SparseBasicRegressionWithClasses(nn.Module):
    """
    Basic sparse CNN model for the regression task
    """
    def __init__(self, image_size: Tuple[int, int, int] = (1, 512, 512),
                 num_classes=4, class_embedding_size=128):
        super(SparseBasicRegressionWithClasses, self).__init__()
        self.image_size = image_size
        algo=spconv.ConvAlgo.MaskImplicitGemm
        self.conv1 = spconv.SparseSequential(
        spconv.SparseConv2d(in_channels=self.image_size[0], out_channels=4,
                            kernel_size=3, padding=1,
                            algo=algo).to(device),
        nn.ReLU(),
        spconv.SparseMaxPool2d(kernel_size=2, stride=2),
        spconv.SubMConv2d(in_channels=4, out_channels=16,
                          kernel_size=3, padding=1,
                          algo=algo,
                          indice_key="subm0").to(device),
        nn.ReLU(),
        spconv.SparseMaxPool2d(kernel_size=2, stride=2),
        spconv.ToDense(),
        )
        
        self.regressor = nn.Sequential(
            nn.LazyLinear(out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=2),
            nn.Tanh() 
        )
        self.class_embedding = nn.Sequential(
            nn.Linear(num_classes, class_embedding_size),
            nn.ReLU(),
            nn.Linear(class_embedding_size, class_embedding_size),
            nn.ReLU(),
        )

    def forward(self, x, class_input):
        # Create sparse tensor
        # you must make a batch axis before call to_sparse
        #torchTensorSp = x.to_sparse() # no channel axis here. equalivant to torchTensor.ndim
        #indices_th = torchTensorSp.indices().permute(1, 0).contiguous().int()
        # sparse tensor features need to have one channel axis.
        #features_th = torchTensorSp.values().view(-1, 1)
        # sparse tensor must have a batch axis, spatial shape dont contain batch axis.
        #x_sp = spconv.SparseConvTensor(features_th, indices_th, x.shape[1:], batch_size)
        x_sp = spconv.SparseConvTensor.from_dense(x.reshape(-1, 512, 512, 1))
        
        # Sparse convolutional layers
        x = self.conv1(x_sp)
        x = x.reshape(x.size(0), -1)
        #  Convert to LongTensor and one-hot encode class inputs
        class_input = F.one_hot(class_input.long() - 1, num_classes=4).float()
        class_input = self.class_embedding(class_input)
        x = torch.cat((x, class_input), dim=1)
        x = self.regressor(x)
        return x
    
class SparseVGG13Regression(nn.Module):
    """
    This is a sparse vgg13 model we will use for the regression task.
    """
    def __init__(self, image_size: Tuple[int, int, int] = (1, 512, 512),
                 num_classes=4, class_embedding_size=128):
        super(SparseVGG13Regression, self).__init__()
        self.image_size = image_size
        algo=spconv.ConvAlgo.MaskImplicitGemm
        self.layer1 = spconv.SparseSequential(
            spconv.SubMConv2d(in_channels=self.image_size[0], out_channels=64,
                              kernel_size=3, padding=1,
                              algo=algo,
                              indice_key="subm0").to(device),
            #nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.layer2 = spconv.SparseSequential(
            spconv.SubMConv2d(in_channels=64, out_channels=64,
                              kernel_size=3, padding=1,
                              algo=algo,
                              indice_key="subm0").to(device),
            #nn.BatchNorm1d(64),
            nn.ReLU(), 
            spconv.SparseMaxPool2d(kernel_size=2, stride=2),
        )
        self.layer3 = spconv.SparseSequential(
            spconv.SubMConv2d(in_channels=64, out_channels=128,
                              kernel_size=3, padding=1,
                              algo=algo,
                              indice_key="subm1").to(device),
            #nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.layer4 = spconv.SparseSequential(
            spconv.SubMConv2d(in_channels=128, out_channels=128,
                              kernel_size=3, padding=1,
                              algo=algo,
                              indice_key="subm1").to(device),
            #nn.BatchNorm1d(128),
            nn.ReLU(),
            spconv.SparseMaxPool2d(kernel_size=2, stride=2),
        )
        self.layer5 = spconv.SparseSequential(
            spconv.SubMConv2d(in_channels=128, out_channels=256,
                              kernel_size=3, padding=1,
                              algo=algo,
                              indice_key="subm2").to(device),
            #nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        self.layer6 = spconv.SparseSequential(
            spconv.SubMConv2d(in_channels=256, out_channels=256,
                              kernel_size=3, padding=1,
                              algo=algo,
                              indice_key="subm2").to(device),
            #nn.BatchNorm1d(256),
            nn.ReLU(),
            spconv.SparseMaxPool2d(kernel_size=2, stride=2),
        )
        self.layer7 = spconv.SparseSequential(
            spconv.SubMConv2d(in_channels=256, out_channels=512,
                              kernel_size=3, padding=1,
                              algo=algo,
                              indice_key="subm3").to(device),
            #nn.BatchNorm1d(512),
            nn.ReLU(),
        )
        self.layer8 = spconv.SparseSequential(
            spconv.SubMConv2d(in_channels=512, out_channels=512,
                              kernel_size=3, padding=1,
                              algo=algo,
                              indice_key="subm3").to(device),
            #nn.BatchNorm1d(512),
            nn.ReLU(),
            spconv.SparseMaxPool2d(kernel_size=2, stride=2),
        )
        self.layer9 = spconv.SparseSequential(
            spconv.SubMConv2d(in_channels=512, out_channels=512,
                              kernel_size=3, padding=1,
                              algo=algo,
                              indice_key="subm4").to(device),
            #nn.BatchNorm1d(4),
            nn.ReLU(),
        )
        self.layer10 = spconv.SparseSequential(
            spconv.SubMConv2d(in_channels=512, out_channels=512,
                              kernel_size=3, padding=1,
                              algo=algo,
                              indice_key="subm4").to(device),
            #nn.BatchNorm1d(4),
            nn.ReLU(),
            spconv.SparseMaxPool2d(kernel_size=2, stride=2),
            spconv.ToDense(),
        )
        
        self.regressor = nn.Sequential(
            nn.LazyLinear(out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=2),
            nn.Tanh() 
        )
        self.class_embedding = nn.Sequential(
            nn.Linear(num_classes, class_embedding_size),
            nn.ReLU(),
            nn.Linear(class_embedding_size, class_embedding_size),
            nn.ReLU(),
        )
          
    def forward(self, x, class_input):
        # Create sparse tensor
        # you must make a batch axis before call to_sparse
        #torchTensorSp = x.to_sparse() # no channel axis here. equalivant to torchTensor.ndim
        #indices_th = torchTensorSp.indices().permute(1, 0).contiguous().int()
        # sparse tensor features need to have one channel axis.
        #features_th = torchTensorSp.values().view(-1, 1)
        # sparse tensor must have a batch axis, spatial shape dont contain batch axis.
        #x_sp = spconv.SparseConvTensor(features_th, indices_th, x.shape[1:], batch_size)
        x_sp = spconv.SparseConvTensor.from_dense(x.reshape(-1, 512, 512, 1))
        
        x = self.layer1(x_sp)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = x.reshape(x.size(0), -1)
        #  Convert to LongTensor and one-hot encode class inputs
        class_input = F.one_hot(class_input.long() - 1, num_classes=4).float()
        class_input = self.class_embedding(class_input)
        x = torch.cat((x, class_input), dim=1)
        x = self.regressor(x)
        return x
    
class SparseVGG16Regression(nn.Module):
    """
    This is a sparse vgg16 model we will use for the regression task.
    """
    def __init__(self, image_size: Tuple[int, int, int] = (1, 512, 512)):
        super(SparseVGG16Regression, self).__init__()
        self.image_size = image_size
        algo=spconv.ConvAlgo.MaskImplicitGemm
        self.layer1 = spconv.SparseSequential(
            spconv.SubMConv2d(in_channels=self.image_size[0], out_channels=64,
                              kernel_size=3, padding=1,
                              algo=algo,
                              indice_key="subm0").to(device),
            #nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.layer2 = spconv.SparseSequential(
            spconv.SubMConv2d(in_channels=64, out_channels=64,
                              kernel_size=3, padding=1,
                              algo=algo,
                              indice_key="subm0").to(device),
            #nn.BatchNorm1d(64),
            nn.ReLU(), 
            spconv.SparseMaxPool2d(kernel_size=2, stride=2),
        )
        self.layer3 = spconv.SparseSequential(
            spconv.SubMConv2d(in_channels=64, out_channels=128,
                              kernel_size=3, padding=1,
                              algo=algo,
                              indice_key="subm1").to(device),
            #nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.layer4 = spconv.SparseSequential(
            spconv.SubMConv2d(in_channels=128, out_channels=128,
                              kernel_size=3, padding=1,
                              algo=algo,
                              indice_key="subm1").to(device),
            #nn.BatchNorm1d(128),
            nn.ReLU(),
            spconv.SparseMaxPool2d(kernel_size=2, stride=2),
        )
        self.layer5 = spconv.SparseSequential(
            spconv.SubMConv2d(in_channels=128, out_channels=256,
                              kernel_size=3, padding=1,
                              algo=algo,
                              indice_key="subm2").to(device),
            #nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        self.layer6 = spconv.SparseSequential(
            spconv.SubMConv2d(in_channels=256, out_channels=256,
                              kernel_size=3, padding=1,
                              algo=algo,
                              indice_key="subm2").to(device),
            #nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        self.layer7 = spconv.SparseSequential(
            spconv.SubMConv2d(in_channels=256, out_channels=256,
                              kernel_size=3, padding=1,
                              algo=algo,
                              indice_key="subm2").to(device),
            #nn.BatchNorm1d(256),
            nn.ReLU(),
            spconv.SparseMaxPool2d(kernel_size=2, stride=2),
        )
        self.layer8 = spconv.SparseSequential(
            spconv.SubMConv2d(in_channels=256, out_channels=512,
                              kernel_size=3, padding=1,
                              algo=algo,
                              indice_key="subm3").to(device),
            #nn.BatchNorm1d(512),
            nn.ReLU(),
        )
        self.layer9 = spconv.SparseSequential(
            spconv.SubMConv2d(in_channels=512, out_channels=512,
                              kernel_size=3, padding=1,
                              algo=algo,
                              indice_key="subm3").to(device),
            #nn.BatchNorm1d(512),
            nn.ReLU(),
        )
        self.layer10 = spconv.SparseSequential(
            spconv.SubMConv2d(in_channels=512, out_channels=512,
                              kernel_size=3, padding=1,
                              algo=algo,
                              indice_key="subm3").to(device),
            #nn.BatchNorm1d(512),
            nn.ReLU(),
            spconv.SparseMaxPool2d(kernel_size=2, stride=2),
        )
        self.layer11 = spconv.SparseSequential(
            spconv.SubMConv2d(in_channels=512, out_channels=512,
                              kernel_size=3, padding=1,
                              algo=algo,
                              indice_key="subm4").to(device),
            #nn.BatchNorm1d(512),
            nn.ReLU(),
        )
        self.layer12 = spconv.SparseSequential(
            spconv.SubMConv2d(in_channels=512, out_channels=512,
                              kernel_size=3, padding=1,
                              algo=algo,
                              indice_key="subm4").to(device),
            #nn.BatchNorm1d(512),
            nn.ReLU(),
        )
        self.layer13 = spconv.SparseSequential(
            spconv.SubMConv2d(in_channels=512, out_channels=512,
                              kernel_size=3, padding=1,
                              algo=algo,
                              indice_key="subm4").to(device),
            #nn.BatchNorm1d(512),
            nn.ReLU(),
            spconv.SparseMaxPool2d(kernel_size=2, stride=2),
            spconv.ToDense(),
        )
        
        self.regressor = nn.Sequential(
            nn.Linear(in_features=1024, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=2)
        )
          
    def forward(self, x):
        # Create sparse tensor
        # you must make a batch axis before call to_sparse
        #torchTensorSp = x.to_sparse() # no channel axis here. equalivant to torchTensor.ndim
        #indices_th = torchTensorSp.indices()[[0,2,3]].permute(1, 0).contiguous().int()
        # sparse tensor features need to have one channel axis.
        #features_th = torchTensorSp.values().view(-1, 1)
        # sparse tensor must have a batch axis, spatial shape dont contain batch axis.
        #x_sp = spconv.SparseConvTensor(features_th, indices_th, x.shape[2:], batch_size)
        x_sp = spconv.SparseConvTensor.from_dense(x.reshape(-1, 512, 512, 1))
        
        x = self.layer1(x_sp)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        x = self.layer12(x)
        x = self.layer13(x)
        x = x.reshape(x.size(0), -1)
        x = self.regressor(x)
        return x
    
def train_network(model, device, n_epochs: int = 10,
                  image_size: Tuple[int, int, int] = (1, 512, 512),
                  batch_size: int = 64, max_angle: float = 1):
    """
    This trains the network for a set number of epochs.
    """
    if image_size[0] == 1:
        grayscale = True
    else:
        grayscale = False
    assert image_size[1] == image_size[2], 'Image size must be square'
    regression_task = RegressionTaskData(grayscale=grayscale, batch_size=batch_size)

    # Define the loss function and optimizer
    model.to(device)
    print(model)
    criterion = nn.MSELoss()
    #optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)
    optimizer = torch.optim.Adamax(model.parameters(), lr=1e-3, weight_decay=0)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
    #factor=0.1, patience=8, threshold=1e-4, threshold_mode='abs', min_lr=1e-14)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=10, T_mult=2,eta_min=1e-14)
    
    # Initialize custom scheduler
    scheduler = CosineAnnealingWarmRestartsDecay(optimizer, T_0=5, T_mult=1, decay=0.7)
    
    # Configure tensorboard layout
    layout = {
        "DSS-CNN: Single-Aperture": {
            "learning rate": ["Multiline", ["lr"]],
            "loss": ["Multiline", ["loss/train", "loss/validation"]],
            "r-squared": ["Multiline", ["r2/train","r2/validation"]],
        },
    }

    # Train the model
    writer = SummaryWriter()
    writer.add_custom_scalars(layout)
    early_stopping = EarlyStopping(patience=6, min_delta=0)
    global_step = 0
    r2_score = R2Score().cpu()
    for epoch in range(n_epochs):
        # Set the model to training mode
        model.train()
        optimizer.zero_grad()  # Initialize gradients to zero
        for i, (inputs, targets, classes) in enumerate(regression_task.trainloader):
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Normalize the targets
            norm_targets = targets/max_angle

            # Forward pass
            outputs = model(inputs.to(device), classes.to(device))
            loss = criterion(outputs, norm_targets.to(device))
            
            # Scale the loss
            scaling_factor = 100  # Example scaling factor
            scaled_loss = scaling_factor * loss
            
            # Compute the R-squared metric
            r2 = r2_score(outputs.view(-1).detach().cpu(), norm_targets.view(-1).detach().cpu())

            # Backward pass and optimization
            scaled_loss.backward() # Backward pass to compute gradients
            optimizer.step()
            scheduler.step(epoch + i / len(regression_task.trainloader))
            
            # Write metrics to tensorboard
            writer.add_scalar("r2/train", r2.item(), global_step=global_step)
            writer.add_scalar("loss/train", loss.item(), global_step=global_step)
            writer.add_scalar("lr", get_lr(optimizer), global_step=global_step)
            
            # Increment global_step
            global_step += 1

            # Print training statistics
            if (i + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{n_epochs}], Step [{i + 1}/{len(regression_task.trainloader)}], R2: {r2.item():.5f}, Loss: {scaled_loss.item():.3E}, LR: {get_lr(optimizer):.3E}')
        # Calculate validation loss
        valid_loss, valid_r2  = validate_network(model, device, criterion,
                                                 r2_score, regression_task.validloader,
                                                 batch_size, max_angle)
        writer.add_scalar("loss/validation", valid_loss.item(), global_step=global_step)
        writer.add_scalar("r2/validation", valid_r2.item(), global_step=global_step)
        # Update learning rate
        #scheduler.step(valid_loss)
        # Check early stop conditions
        early_stopping(valid_loss)
        if early_stopping.early_stop:
            break
    writer.close()

    return model

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True

def save_model(model, filename='1_512_512.pth'):
    """
    After training the model, save it so we can use it later.
    """
    torch.save(model.state_dict(), filename)


def load_model(model, image_size=(1, 512, 512), filename='1_512_512.pth'):
    """
    Load the model from the saved state dictionary.
    """
    model.load_state_dict(torch.load(filename))
    return model

def validate_network(model, device, criterion, r2_score, valid_loader,
                     batch_size: int = 64, max_angle: float = 1):
    """
    This validates the network on the valid data.
    """
    # Validate the model on the valid data
    with torch.no_grad():
        # Set the model to evaluation mode
        model.eval()
        for inputs, targets, classes in valid_loader:
            # Normalize the targets
            norm_targets = targets/max_angle
            # Calculate the loss with the criterion we used in validation
            outputs = model(inputs.to(device), classes.to(device))
            loss = criterion(outputs, norm_targets.to(device))
            # Scale the loss
            scaling_factor = 1  # Example scaling factor
            scaled_loss = scaling_factor * loss
            # Compute the R-squared metric
            r2 = r2_score(outputs.view(-1).detach().cpu(), norm_targets.view(-1).detach().cpu())
    return scaled_loss, r2

def generate_error_image_alpha(error_image, model, regression_task_set, device, max_angle):
    # Initialize train vectors A
    output_a_vec = np.array([])
    target_a_vec = np.array([])
    error_a_vec = np.array([])
    
    # Initialize train vectors B
    target_b_vec = np.array([])

    for inputs, targets, classes in regression_task_set:
        outputs = model(inputs.to(device), classes.to(device))

        outputs_np = outputs.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()

        output_a = np.array([out[0] for out in outputs_np])
        target_a = np.array([t[0] for t in targets_np])
        
        target_b = np.array([t[1] for t in targets_np])
        
        error_a = np.abs(target_a - output_a)

        # Update train vectors A
        output_a_vec = np.concatenate((output_a_vec, output_a))
        target_a_vec = np.concatenate((target_a_vec, target_a))
        error_a_vec = np.concatenate((error_a_vec, error_a))
        
        # Update train vectors B
        target_b_vec = np.concatenate((target_b_vec, target_b))
    
    # Create the error image
    pix_x = np.round(5000 / 1.55 * np.tan(target_a_vec)) + 256
    pix_y = np.round(5000 / 1.55 * np.tan(target_b_vec)) + 256
    for x, y, value in zip(pix_x.astype(int), pix_y.astype(int), error_a_vec*180/pi*max_angle):
            error_image[y, x] = value

    return error_image

def generate_error_image_beta(error_image, model, regression_task_set, device, max_angle):
    # Initialize train vectors A
    target_a_vec = np.array([])
    
    # Initialize train vectors B
    output_b_vec = np.array([])
    target_b_vec = np.array([])
    error_b_vec = np.array([])

    for inputs, targets, classes in regression_task_set:
        outputs = model(inputs.to(device), classes.to(device))

        outputs_np = outputs.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()

        target_a = np.array([t[0] for t in targets_np])
        
        output_b = np.array([out[1] for out in outputs_np])
        target_b = np.array([t[1] for t in targets_np])
        
        error_b = np.abs(target_b - output_b)

        # Update train vectors A
        target_a_vec = np.concatenate((target_a_vec, target_a))
        
        # Update train vectors B
        output_b_vec = np.concatenate((output_b_vec, output_b))
        target_b_vec = np.concatenate((target_b_vec, target_b))
        error_b_vec = np.concatenate((error_b_vec, error_b))
    
    # Create the error image
    pix_x = np.round(5000 / 1.55 * np.tan(target_a_vec)) + 256
    pix_y = np.round(5000 / 1.55 * np.tan(target_b_vec)) + 256
    for x, y, value in zip(pix_x.astype(int), pix_y.astype(int), error_b_vec*180/pi*max_angle):
            error_image[y, x] = value

    return error_image

def make_error_image_alpha(model, regression_task, device, max_angle):
    # Initialize error image
    error_image_size = (513, 513)
    error_image = np.empty(error_image_size)
    error_image[:] = np.nan

    # Create test set error image
    error_image = generate_error_image_alpha(error_image, model, regression_task.testloader, device, max_angle)
    # Create train set error image
    error_image = generate_error_image_alpha(error_image, model, regression_task.trainloader, device, max_angle)
    # Create validation set error image
    error_image = generate_error_image_alpha(error_image, model, regression_task.validloader, device, max_angle)

    return error_image

def make_error_image_beta(model, regression_task, device, max_angle):
    # Initialize error image
    error_image_size = (513, 513)
    error_image = np.empty(error_image_size)
    error_image[:] = np.nan

    # Create test set error image
    error_image = generate_error_image_beta(error_image, model, regression_task.testloader, device, max_angle)
    # Create train set error image
    error_image = generate_error_image_beta(error_image, model, regression_task.trainloader, device, max_angle)
    # Create validation set error image
    error_image = generate_error_image_beta(error_image, model, regression_task.validloader, device, max_angle)

    return error_image
        
def evaluate_network(model, device,
                     image_size: Tuple[int, int, int] = (1, 512, 512),
                     batch_size: int = 64, max_angle: float = 1):
    """
    This evaluates the network on the test data.
    """
    if image_size[0] == 1:
        grayscale = True
    else:
        grayscale = False
    assert image_size[1] == image_size[2], 'Image size must be square'
    regression_task = RegressionTaskData(grayscale=grayscale, batch_size=batch_size)
    criterion = nn.MSELoss()

    # Evaluate the model on the test data
    with torch.no_grad():
        # Initialize metrics
        total_loss = 0
        total_a_error = 0
        total_b_error = 0
        n_samples_total = 0
        
        # Initialize test vectors A
        output_a_vec = np.array([])
        target_a_vec = np.array([])
        error_a_vec = np.array([])
        
        # Initialize test vectors B
        output_b_vec = np.array([])
        target_b_vec = np.array([])
        error_b_vec = np.array([])
        
        for inputs, targets, classes in regression_task.testloader:
            
            # Normalize the targets
            norm_targets = targets/max_angle
            # Calculate the loss with the criterion we used in training
            outputs = model(inputs.to(device), classes.to(device))
            loss = criterion(outputs, norm_targets.to(device))
            total_loss += loss.item()

            outputs_np = outputs.detach().cpu().numpy()
            targets_np = norm_targets.detach().cpu().numpy()

            output_a = np.array([out[0] for out in outputs_np])
            target_a = np.array([t[0] for t in targets_np])
            
            output_b = np.array([out[1] for out in outputs_np])
            target_b = np.array([t[1] for t in targets_np])
            
            error_a = np.abs(target_a - output_a)
            error_b = np.abs(target_b - output_b)
            
            error_a_sum = np.sum(error_a)
            error_b_sum = np.sum(error_b)
            
            total_a_error += error_a_sum
            total_b_error += error_b_sum

            n_samples_total += len(output_a)
            
            # Update test vectors A
            output_a_vec = np.concatenate((output_a_vec, output_a))
            target_a_vec = np.concatenate((target_a_vec, target_a))
            error_a_vec = np.concatenate((error_a_vec, error_a))
            
            # Update test vectors B
            output_b_vec = np.concatenate((output_b_vec, output_b))
            target_b_vec = np.concatenate((target_b_vec, target_b))
            error_b_vec = np.concatenate((error_b_vec, error_b))

        mean_loss = total_loss / len(regression_task.testloader)
        mean_a_error = total_a_error / n_samples_total *180/pi*max_angle
        mean_b_error = total_b_error / n_samples_total *180/pi*max_angle
        print(f'Test Loss: {mean_loss:.3E}')
        print(f'Test mean alpha error: {mean_a_error:.4f} deg')
        print(f'Test mean beta error: {mean_b_error:.4f} deg')
        
        # Plot the error histogram for test A
        # Proposed method
        sns.histplot(error_a_vec*180/pi*max_angle, bins=100, stat='probability', kde=True, zorder=2)
        plt.xlabel('Absolute Error $\\alpha$ Angle (deg)')
        plt.ylabel('Probability')
        plt.show()
        
        # Plot the error histogram for test B
        # Proposed method
        sns.histplot(error_b_vec*180/pi*max_angle, bins=100, stat='probability', kde=True, zorder=2)
        plt.xlabel('Absolute Error $\\beta$ Angle (deg)')
        plt.ylabel('Probability')
        plt.show()
        
        # Plot the predicted vs actual for test A
        xtrue = np.linspace(-0.1,0.1,512)
        ytrue = xtrue
        plt.plot(xtrue, ytrue, linewidth=1, color='red', zorder=2)
        plt.scatter(output_a_vec*max_angle, target_a_vec*max_angle, s=10, c=error_a_vec, cmap='viridis', alpha=0.5, zorder=1)
        plt.xlabel('Actual $\\alpha$ Angle')
        plt.ylabel('Predicted $\\alpha$ Angle')
        #plt.title('Actual vs Predicted X Vector')
        plt.colorbar()
        plt.show()
        
        # Plot the predicted vs actual for test B
        xtrue = np.linspace(-0.1,0.1,512)
        ytrue = xtrue
        plt.plot(xtrue, ytrue, linewidth=1, color='red', zorder=2)
        plt.scatter(output_b_vec*max_angle, target_b_vec*max_angle, s=10, facecolors='none', c=error_b_vec, cmap='viridis', alpha=0.5, zorder=1)
        plt.xlabel('Actual $\\beta$ Angle')
        plt.ylabel('Predicted $\\beta$ Angle')
        #plt.title('Actual vs Predicted Y Vector')
        plt.colorbar()
        plt.show()
        
        # Save the error histogram as a csv file
        np.savetxt('sa_aug_hist_alpha.csv', error_a_vec*180/pi*max_angle, delimiter=',')
        np.savetxt('sa_aug_hist_beta.csv', error_b_vec*180/pi*max_angle, delimiter=',')
        
        # Save the actvspred as a csv file
        actvspred_a = np.vstack((output_a_vec*max_angle, target_a_vec*max_angle, error_a_vec))
        actvspred_b = np.vstack((output_b_vec*max_angle, target_b_vec*max_angle, error_b_vec))
        np.savetxt('sa_aug_actvspred_alpha.csv', actvspred_a.T, delimiter=',',
                   header='Output,Target,Error', comments='') 
        np.savetxt('sa_aug_actvspred_beta.csv', actvspred_b.T, delimiter=',',
                   header='Output,Target,Error', comments='') 
        
        # Create the error image
        #error_image_alpha = make_error_image_alpha(model, regression_task, device, max_angle)
        #error_image_beta = make_error_image_beta(model, regression_task, device, max_angle)
        
        # Save the error image as a csv file
        #np.savetxt('sa_noaug_errorimage_alpha.csv', error_image_alpha, delimiter=',')
        #np.savetxt('sa_noaug_errorimage_beta.csv', error_image_beta, delimiter=',')
        
        # Load the error image as a csv file
        #error_image_alpha = np.loadtxt('sa_noaug_errorimage_alpha.csv', delimiter=',')
        #error_image_beta = np.loadtxt('sa_noaug_errorimage_beta.csv', delimiter=',')
        
        # Plot the error images
        #plt.imshow(error_image_alpha, cmap='viridis', vmin=0, vmax=0.05)
        #plt.xlabel('$\\alpha$ Angle (deg)')
        #plt.ylabel('$\\beta$ Angle (deg)')
        #cbar = plt.colorbar()
        #cbar.set_label('Absolute Error $\\alpha$ Angle (deg)')
        #plt.show()
        
        #plt.imshow(error_image_beta, cmap='viridis', vmin=0, vmax=0.05)
        #plt.xlabel('$\\alpha$ Angle (deg)')
        #plt.ylabel('$\\beta$ Angle (deg)')
        #cbar = plt.colorbar()
        #cbar.set_label('Absolute Error $\\beta$ Angle (deg)')
        #plt.show()
        

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    
    # Define max angle
    #max_angle = 0.079194
    max_angle = 1
    
    # Define training parameters and model
    image_size: Tuple[int, int, int] = (1, 512, 512)
    n_epochs = 100
    batch_size = 64
    #model_select = ResNet(ResidualBlock, [3, 4, 6, 3])
    model_select = SparseResNet34(SparseBasicBlock, [3, 4, 6, 3], num_classes=4, class_embedding_size=2048)
    #model_select = SparseVGG13Regression(image_size=image_size,num_classes=4,class_embedding_size=2048)
    #model_select = SparseBasicRegression(image_size=image_size)
    
    # Train the model
    #model = train_network(model_select, device, n_epochs, image_size=image_size,
    #                      batch_size=batch_size, max_angle=max_angle)

    # Save the model
    filename = 'sa_aug.pth'
    #filename = f'{image_size[0]}_{image_size[1]}_{image_size[2]}.pth'
    #save_model(model, filename=filename)

    # Load the model
    #filename = 'sa_aug.pth'
    model = load_model(model_select, image_size=image_size, filename=filename)
    model.to(device)

    # Evaluate the model
    evaluate_network(model, device, image_size=image_size,
                     batch_size=batch_size, max_angle=max_angle)