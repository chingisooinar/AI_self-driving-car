#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 11:31:13 2021

@author: chingis
"""

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
import math
from .MotionEncoder import TransformerMotionEncoder, TransformerMotionEncoderLayer
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class MotionTransformer(nn.Module):
    def __init__(self, seq_len):
        self.seq_len = seq_len
        super(MotionTransformer,self).__init__()
        self.position_encoder = models.resnet18(pretrained=True)
        self.motion_encoder = models.resnet18(pretrained=True)
        self.d_model = 512
        self.position_embedder = nn.Linear(in_features =1000, out_features = self.d_model, bias=True)
        self.motion_embedder = nn.Linear(in_features =1000, out_features = self.d_model, bias=True)
        
        self.encoder_layer = TransformerMotionEncoderLayer(d_model=self.d_model, nhead=4, dropout=0.1)
        self.pos_encoder = PositionalEncoding(d_model=512)
        self.transformer_encoder = TransformerMotionEncoder(self.encoder_layer, num_layers=2, norm=None) #nn.LayerNorm(512)
        

        
        self.reduce_combined = nn.Linear(in_features =self.d_model, out_features = 64, bias=True)
        
        #self.reduce = nn.Linear(in_features = 128, out_features = 64, bias=True)
        self.steering_predictor = nn.Linear(in_features = 64, out_features = 1, bias=True)

        #self.motion_retrieval = nn.Linear(in_features = 128, out_features = 64, bias=True)
        self.speed_predictor = nn.Linear(in_features = 64, out_features = 1, bias=True)
    
    def generate_square_subsequent_mask(self,sz: int):
        """Generates an upper-triangular matrix of -inf, with zeros on diag."""
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
    
    def forward(self, frames, optical):
        frames = frames.reshape(-1, 3, 224, 224)
        optical = optical.reshape(-1, 3, 224, 224)
       # print(x.shape)
        frames = F.relu(self.position_embedder(self.position_encoder(frames)))
        frames = frames.reshape(-1,  self.seq_len, self.d_model).permute(1, 0, 2)


        optical = F.relu(self.motion_embedder(self.motion_encoder(optical)))
        optical = optical.reshape(-1,  self.seq_len, self.d_model).permute(1, 0, 2)
        

        #frames = self.pos_encoder(frames)
        attn_mask = self.generate_square_subsequent_mask(frames.shape[0]).cuda()
        fused_embedding = F.relu(self.transformer_encoder((frames, optical), mask=attn_mask))
        # LSTM 16
        #image = torch.tanh(self.LSTM2(image)[0])

        fused_embedding = fused_embedding.permute(1, 0, 2)
        fused_embedding = fused_embedding.reshape(-1, self.d_model)
        reduced = F.relu(self.reduce_combined(fused_embedding))
        #reduced = F.relu(self.reduce(reduced))
        #motion_information = F.relu(self.motion_retrieval(reduced))

        # FC 1
        speed = self.speed_predictor(reduced)
 
        # FC 16
        #steering_information = F.relu(self.steering_retrieval(reduced))

        # FC 1
        angle = self.steering_predictor(reduced)

        return angle, speed
