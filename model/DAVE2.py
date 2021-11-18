#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 13:52:33 2021

@author: chingis
"""
import torch
import torch.nn as nn
class DAVE2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers=nn.Sequential(
            nn.Conv2d(3,24,5,stride=2),
            nn.ELU(),
            nn.Conv2d(24,36,5,stride=2),
            nn.ELU(),
            nn.Conv2d(36,48,5, stride=2),
            nn.ELU(),
            nn.Conv2d(48, 64, 3),
            nn.ELU(),
            nn.Conv2d(64, 64, 3),
            nn.Dropout(0.5)   
        )
        self.dense_layers=nn.Sequential(
            nn.Linear(in_features=64 * 8 * 33, out_features=100),
            nn.ELU(),
            nn.Linear(in_features=100, out_features=50),
            nn.ELU(),
            nn.Linear(in_features=50, out_features=10),
            nn.ELU(),
            nn.Linear(in_features=10, out_features=1)
        )
    def forward(self,data):
              
        output = self.conv_layers(data)
        output = output.view(output.size(0), -1)
        output = self.dense_layers(output)
        return output

if __name__ == '__main__':
    car = DAVE2()
    x = torch.randn(32, 3, 120, 320)