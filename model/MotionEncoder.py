#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 10:38:08 2021

@author: chingis
"""
import torch
import torch.nn as nn
import math
import copy


class TransformerMotionEncoderLayer(nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.25, activation="relu"):
        super(TransformerMotionEncoderLayer, self).__init__()
        self.self_attn_frames = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn_motion = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear1_m = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.dropout_m = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.linear2_m = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm1_m = nn.LayerNorm(d_model)
        self.norm2_m = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout1_m = nn.Dropout(dropout)
        self.dropout2_m = nn.Dropout(dropout)
        self.activation = nn.ReLU() if activation == 'relu' else nn.LeakyReLU(0.2)
        self.d_model = d_model
        


    def forward(self, frames_embeddings, motion_embeddings, src_mask=None, src_key_padding_mask=None):
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        
        # skip connection for frames
        frame_attention = self.self_attn_frames(frames_embeddings, frames_embeddings, frames_embeddings, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        frames_embeddings = frames_embeddings + self.dropout1(frame_attention)
        src = self.norm1(frames_embeddings)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(frames_embeddings))))
        src = src + self.dropout2(src2)
        src_frames = self.norm2(src)
        # skip connection for motion
        
        
        
        motion_attention = self.self_attn_frames(frames_embeddings, frames_embeddings, motion_embeddings, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        motion_embeddings = motion_embeddings + self.dropout1_m(motion_attention)
        src = self.norm1_m(motion_embeddings)
        src2 = self.linear2_m(self.dropout_m(self.activation(self.linear1_m(motion_embeddings))))
        src = src + self.dropout2_m(src2)
        src_motion = self.norm2_m(src)
        assert src_motion.shape == src_frames.shape
        
        return src_frames, src_motion

class TransformerMotionEncoder(nn.Module):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer,  num_layers, norm=None):
        super(TransformerMotionEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.attention_linear = nn.Linear(encoder_layer.d_model * 2, encoder_layer.d_model)
    
    def forward(self, src, mask=None, src_key_padding_mask=None):
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src

        for mod in self.layers:
            output = mod(*output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        x = torch.cat(output, dim=-1)
        output = self.attention_linear(x)
        if self.norm is not None:
            output = self.norm(output)

        return output
    
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])