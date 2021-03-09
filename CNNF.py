# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 10:37:12 2021

@author: Hanan
"""

# Importing Libraries

import torch
import torch.nn as nn
from CNN1 import CNN1
from CNN2 import CNN2

# Creating CNNF Model

class CNNF(nn.Module):
    def __init__(self):
        super().__init__()
        self.Block1 = nn.Sequential(nn.Linear(8*8*1536 , 5), nn.Softmax(dim=1))
        self.model1 = CNN1(chanells=1, fc_activation ='relu')
        self.model2 = CNN2(fc_activation ='relu')
        
    def forward(self,x,y):
        output = torch.cat((self.model1(x), self.model2(y)), dim=1)
        output = self.Block1(output)
        return output
