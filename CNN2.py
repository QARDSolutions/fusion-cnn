# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 10:36:01 2021

@author: Hanan
"""

# Importing Libraries

import torch
import torch.nn as nn
from denseunet import UNet

# Creating CNN2 Model
class CNN2(nn.Module):
    
    def __init__(self, fc_activation = 'softmax'):
        
        self.fc_activation = fc_activation
        super().__init__()
        
        # No. of Channels ,  No. of filters , Filter size
        
        # Block 1
        self.Block1 = nn.Sequential(
            nn.Conv2d(1 , 64 , 1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2)
            )
        
        # Block 2
        self.Block2 = nn.Sequential(
            nn.Conv2d(64 , 128 , 1),
            nn.LeakyReLU(),
            nn.Conv2d(128 , 128 , 1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2)
            )
        
        # Block 3
        self.Block3 = nn.Sequential(
            nn.Conv2d(128 , 256 , 1),
            nn.LeakyReLU(),
            nn.Conv2d(256 , 256 , 1),
            nn.LeakyReLU(),
            nn.Conv2d(256 , 256 , 1),
            nn.LeakyReLU(),
            nn.Conv2d(256 , 256 , 1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2)
            )
        
        # Block 4
        self.Block4 = nn.Sequential(
            nn.Conv2d(256 , 384 , 1),
            nn.LeakyReLU(),
            nn.Conv2d(384 , 384 , 1),
            nn.LeakyReLU(),
            nn.Conv2d(384 , 384 , 1),
            nn.LeakyReLU(),
            nn.Conv2d(384 , 384 , 1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2)
            )
        
        # Block 5
        self.Block5 = nn.Sequential(
            nn.Conv2d(384 , 512 , 1),
            nn.LeakyReLU(),
            nn.Conv2d(512 , 512 , 1),
            nn.LeakyReLU(),
            nn.Conv2d(512 , 512 , 1),
            nn.LeakyReLU(),
            nn.Conv2d(512 , 512 , 1),
            nn.LeakyReLU(),
            nn.AvgPool2d(2 , 2)
            )
        
        # Block 6
        self.Block6 = nn.Sequential(
                nn.Linear(8*8*512 , 5),
                nn.Softmax(dim=1)
                )
        #UNET
        #self.denseUnet = UNet(in_channels=1, out_channels=1, n_blocks=4, start_filters=32,
        #     activation='relu', normalization='batch', conv_mode='same', dim=2)

    
    def forward(self, x):
        #x = self.denseUnet(x)
        x = self.Block1(x)
        x = self.Block2(x)
        x = self.Block3(x)
        x = self.Block4(x)
        x = self.Block5(x)
        x = x.view(-1, 512*8*8)
        
        if self.fc_activation == 'softmax':
            x = self.Block6(x)
        
        return x
