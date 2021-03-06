# Importing Libraries

import torch
import torch.nn as nn

# Creating CNN Models

class CNN1(nn.Module):
    
    def __init__(self , chanells):
        
        self.chanells = chanells
        super().__init__()
        
        # No. of Channels ,  No. of filters , Filter size
        
        # Block 1
        self.Block1 = nn.Sequential(
            nn.Conv2d(self.chanells , 64 , 1),
            nn.MaxPool2d(2, 2)
            )
        
        # Block 2
        self.Block2 = nn.Sequential(
            nn.Conv2d(64 , 128 , 1),
            nn.Conv2d(128 , 128 , 1),
            nn.MaxPool2d(2, 2)
            )
        
        # Block 3
        self.Block3 = nn.Sequential(
            nn.Conv2d(128 , 256 , 1),
            nn.Conv2d(256 , 256 , 1),
            nn.Conv2d(256 , 256 , 1),
            nn.Conv2d(256 , 256 , 1),
            nn.MaxPool2d(2, 2)
            )
        
        # Block 4
        self.Block4 = nn.Sequential(
            nn.Conv2d(256 , 512 , 1),
            nn.Conv2d(512 , 512 , 1),
            nn.Conv2d(512 , 512 , 1),
            nn.Conv2d(512 , 512 , 1),
            nn.Conv2d(512 , 512 , 1),
            nn.Conv2d(512 , 512 , 1),
            nn.Conv2d(512 , 512 , 1),
            nn.Conv2d(512 , 512 , 1),
            nn.MaxPool2d(2, 2)
            )
        
        # Block 5
        self.Block5 = nn.Sequential(
            nn.Conv2d(512 , 1024 , 1),
            nn.Conv2d(1024 , 1024 , 1),
            nn.Conv2d(1024 , 1024 , 1),
            nn.Conv2d(1024 , 1024 , 1),
            nn.Conv2d(1024 , 1024 , 1),
            nn.Conv2d(1024 , 1024 , 1),
            nn.Conv2d(1024 , 1024 , 1),
            nn.Conv2d(1024 , 1024 , 1),
            nn.Conv2d(1024 , 1024 , 1),
            nn.Conv2d(1024 , 1024 , 1),
            nn.Conv2d(1024 , 1024 , 1),
            nn.Conv2d(1024 , 1024 , 1),
            nn.Conv2d(1024 , 1024 , 1),
            nn.Conv2d(1024 , 1024 , 1),
            nn.Conv2d(1024 , 1024 , 1),
            nn.Conv2d(1024 , 1024 , 1),
            nn.AvgPool2d(2 , 2)
            )
        
        # Block 6
        self.Block6 = nn.Sequential(
            nn.Linear(8*8*1024 , 2)
            )
    
    def forward(self, x):
        x = self.Block1(x)
        x = self.Block2(x)
        x = self.Block3(x)
        x = self.Block4(x)
        x = self.Block5(x)
        x = x.view(-1, 1024*8*8)
        #x = self.Block6(x)
        
        return x


class CNN2(nn.Module):
    
    def __init__(self):
        
        super().__init__()
        
        # No. of Channels ,  No. of filters , Filter size
        
        # Block 1
        self.Block1 = nn.Sequential(
            nn.Conv2d(3 , 64 , 1),
            nn.MaxPool2d(2, 2)
            )
        
        # Block 2
        self.Block2 = nn.Sequential(
            nn.Conv2d(64 , 128 , 1),
            nn.Conv2d(128 , 128 , 1),
            nn.MaxPool2d(2, 2)
            )
        
        # Block 3
        self.Block3 = nn.Sequential(
            nn.Conv2d(128 , 256 , 1),
            nn.Conv2d(256 , 256 , 1),
            nn.Conv2d(256 , 256 , 1),
            nn.Conv2d(256 , 256 , 1),
            nn.MaxPool2d(2, 2)
            )
        
        # Block 4
        self.Block4 = nn.Sequential(
            nn.Conv2d(256 , 384 , 1),
            nn.Conv2d(384 , 384 , 1),
            nn.Conv2d(384 , 384 , 1),
            nn.Conv2d(384 , 384 , 1),
            nn.MaxPool2d(2, 2)
            )
        
        # Block 5
        self.Block5 = nn.Sequential(
            nn.Conv2d(384 , 512 , 1),
            nn.Conv2d(512 , 512 , 1),
            nn.Conv2d(512 , 512 , 1),
            nn.Conv2d(512 , 512 , 1),
            nn.AvgPool2d(2 , 2)
            )
        
        # Block 6
        self.Block6 = nn.Sequential(
            nn.Linear(8*8*512 , 2)
            )
    
    def forward(self, x):
        x = self.Block1(x)
        x = self.Block2(x)
        x = self.Block3(x)
        x = self.Block4(x)
        x = self.Block5(x)
        x = x.view(-1, 512*8*8)
        #x = self.Block6(x)
        
        return x
    
class CNNF(nn.Module):
    
    def __init__(self):
        
        super().__init__()
        
        self.Block1 = nn.Sequential(
            nn.Linear(8*8*1536 , 2)
            )
    
    def forward(self,x,y):
        model1 = CNN1()
        model2 = CNN2()
        output = torch.cat ((model1(x) , model2(y)) , dim=1)
        output = self.Block1(output)
        
        return output
    
# Testing the Models
        
def test():
    model1 = CNNF()
    x = torch.randn((1,3,256,256)) # Original Image
    y = torch.randn((1,3,256,256)) # ROI Image
    print (model1(x,y).shape)

test()