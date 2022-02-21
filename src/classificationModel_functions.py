# -*- coding: utf-8 -*-
"""
@author: DeepCaT_Z
"""

import torch
from torch import nn
from sklearn.metrics import f1_score

from config_DeepCaT_Z import *

#%% UTILS

class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return (x.contiguous()).view(self.shape)

class SliceRNN(nn.Module):
    def forward(self, x):
        output, hn = x
        return output[:, -1]

  
#%% Classification Model 

class Model_X_A(nn.Module):
    def __init__(self, timesteps):
        super().__init__()
        
        # 128 => 64, 8, 8   (4096)
        # 256 => 16, 16, 16 (4096)
        # 512 => 4, 32, 32  (4096)        
        last_kernel = {32: 1024, 64: 256, 128: 64, 256: 16, 512: 4}
        
        self.net = nn.Sequential(
            Reshape(-1, CHANNELS_SIZE, IMG_SIZE, IMG_SIZE),
            nn.Conv2d(CHANNELS_SIZE, 64, 3, 2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 2, padding=1), nn.ReLU(),
            nn.Conv2d(64, last_kernel[IMG_SIZE], 3, 2, padding=1), nn.ReLU(),
            
            Reshape(-1, timesteps, 64*8*8),
            nn.Dropout(),
            
            nn.RNN(64*8*8, 128, batch_first= True),
            SliceRNN(),
            
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, NCLASSES),
        )

    def forward(self, x):
        return (self.net(x),)
