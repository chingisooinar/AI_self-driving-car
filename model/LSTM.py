#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 13:44:16 2021

@author: chingis
"""

import torch
from torchvision import transforms, utils
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
class SequenceModel(nn.Module):
    def __init__(self, seq_len):
        self.seq_len = seq_len
        super(SequenceModel,self).__init__()
        self.ResNet = models.resnet50(pretrained=True)
        
        self.fc0 = nn.Linear(in_features =1000, out_features = 512, bias=True)
        
        self.LSTM1 = nn.LSTM(input_size = 512, hidden_size = 256, num_layers = 2, batch_first=True)
        #self.LSTM2 = nn.LSTM(input_size = 256, hidden_size = 256, num_layers = 1, batch_first=True)
        
        self.fc1 = nn.Linear(in_features =256, out_features = 64, bias=True)
        #self.fc2 = nn.Linear(in_features = 128, out_features = 128, bias=True)
        #self.fc3 = nn.Linear(in_features = 128, out_features = 64, bias=True)
        self.fc4 = nn.Linear(in_features = 64, out_features = 16, bias=True)
        self.fc5 = nn.Linear(in_features = 16, out_features = 1, bias=True)
        self.dropout = nn.Dropout(0.5)
    def forward(self, Input):
        x = Input.reshape(-1, 3, 224, 224)
       # print(x.shape)
        x = F.relu(self.fc0(self.ResNet(x)))
        image = x.reshape(-1,  self.seq_len, 512)
       # print(image.shape)
        h = torch.zeros(2, image.shape[0], 256).cuda()
        c = torch.zeros(2, image.shape[0], 256).cuda()
        image = self.LSTM1(image, (h,c))
        image = image[0]
        image = torch.tanh(image)
        
        # LSTM 16
        #image = torch.tanh(self.LSTM2(image)[0])

        # FC 512
        image = image.reshape(-1, 256)
        image = F.relu(self.fc1(image))
 
        # FC 128
        #image = F.relu(self.fc2(image))

        # FC 64
        #image = F.relu(self.fc3(image))
 
        # FC 16
        image = F.relu(self.fc4(image))

        # FC 1
        angle = self.fc5(image)

        return angle