"""
This file includes the Conv Net functions and classes
that will be imported into the main network code.

Rory Farwell and Patrick Hastings 22/03/2022
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):

    def __init__(self):
      super(CNN, self).__init__()
      self.conv1 = nn.Conv3d(1,32,2,2)
      self.pool = nn.MaxPool3d(2,2)
      self.avg_pool = nn.AvgPool3d(10)
      self.conv2 = nn.Conv3d(32,128,2,2)
      self.conv3 = nn.Conv3d(128,64,1,1)
      self.conv4 = nn.Conv3d(64,16,1,1)
      self.conv5 = nn.Conv3d(16,2,1,1)

    # Defining the forward pass  (NIN method)  
    def forward(self, x):
        x = self.pool(F.leaky_relu(self.conv1(x)))
        x = self.pool(F.leaky_relu(self.conv2(x)))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x = self.avg_pool(self.conv5(x))
        x = x.view(-1,2)
        return x