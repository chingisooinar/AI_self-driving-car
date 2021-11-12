#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 12:24:40 2021

@author: chingis
"""

from skimage.transform import resize
from itertools import product
from easydict import EasyDict as edict
from tqdm import tqdm
import torch.backends.cudnn as cudnn
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, utils, models
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from DataLoading import UdacityDataset as UD
from DataLoading import ConsecutiveBatchSampler as CB

from model import Convolution3D as CNN3D
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb
# noinspection PyAttributeOutsideInit
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
device = torch.device("cuda")

parameters = edict(
    file = '3DCNN_Paper', # used to mark specific files in case that we want to check them on tensorboard
    learning_rate = 0.001,
    batch_size = 32,
    seq_len = 5,
    num_workers = 8,
)


network = CNN3D.Convolution3D().to(device)


wandb.init(config=parameters, project='self-driving-car')
wandb.watch(network)
optimizer = optim.Adam(network.parameters(),lr = parameters.learning_rate,betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001)

udacity_dataset = UD.UdacityDataset(csv_file='/home/chingis/self-driving-car/output/interpolated.csv',
                             root_dir='/home/chingis/self-driving-car/output/',
                             transform=transforms.Compose([transforms.ToTensor()]),
                             select_camera='center_camera')

dataset_size = int(len(udacity_dataset))
del udacity_dataset
split_point = int(dataset_size * 0.9)

training_set = UD.UdacityDataset(csv_file='/home/chingis/self-driving-car/output/interpolated.csv',
                             root_dir='/home/chingis/self-driving-car/output/',
                             transform=transforms.Compose([
                                 transforms.ToTensor()]),
                             select_camera='center_camera',
                             select_range=(0,split_point))
validation_set = UD.UdacityDataset(csv_file='/home/chingis/self-driving-car/output/interpolated.csv',
                             root_dir='/home/chingis/self-driving-car/output/',
                             transform=transforms.Compose([transforms.ToTensor()]),
                             select_camera='center_camera',
                             select_range=(split_point,dataset_size))

training_cbs = CB.ConsecutiveBatchSampler(data_source=training_set, batch_size=parameters.batch_size, shuffle=False, drop_last=False, seq_len=parameters.seq_len)
training_loader = DataLoader(training_set, sampler=training_cbs, num_workers=parameters.num_workers, collate_fn=(lambda x: x[0]))

validation_cbs = CB.ConsecutiveBatchSampler(data_source=validation_set, batch_size=parameters.batch_size, shuffle=False, drop_last=False, seq_len=parameters.seq_len)
validation_loader = DataLoader(validation_set, sampler=validation_cbs, num_workers=parameters.num_workers, collate_fn=(lambda x: x[0]))
criterion =  torch.nn.MSELoss()
criterion.to(device)
for epoch in range(32):
    losses = AverageMeter()
    torch.save(network.state_dict(), "saved_models/CNN3D/epoch-{}.tar".format(epoch))
    network.train()
    # Calculation on Training Loss
    for training_sample in tqdm(training_loader):
        training_sample['image'] = torch.Tensor(resize(training_sample['image'], (parameters.batch_size,parameters.seq_len,3,120,320),anti_aliasing=True))
        training_sample['image'] = training_sample['image'].permute(0,2,1,3,4)
        
        param_values = [v for v in training_sample.values()]
        image,angle = param_values[0],param_values[3]
        #print(image.shape)

        image_flipped = torch.fliplr(image)
        angle_flipped = angle * -1.0
        image = torch.cat((image, image_flipped))
        angle = torch.cat((angle, angle_flipped))
        
        image = image.to(device)
        prediction = network(image)
        #print(prediction.shape)
        prediction = prediction.reshape(parameters.batch_size * 2,parameters.seq_len)
        labels = angle.float().reshape(parameters.batch_size * 2,parameters.seq_len).to(device)

        if labels.shape[0]!=prediction.shape[0]:
            prediction = prediction[-labels.shape[0],:]
        #print(prediction[0], labels[0])
        training_loss_angle =criterion(prediction,labels)
        #print(training_loss_angle)
        losses.update(training_loss_angle.item())
        optimizer.zero_grad()# zero the gradient that are being held in the Grad attribute of the weights
        training_loss_angle.backward() # calculate the gradients
        optimizer.step() # finishing calculation on gradient
    
    print("Done")
    network.eval()
    # Calculation on Validation Loss
    val_losses = AverageMeter()
    with torch.no_grad():    
        for Validation_sample in tqdm(validation_loader):
            Validation_sample['image'] = torch.Tensor(resize(Validation_sample['image'], (parameters.batch_size,parameters.seq_len,3,120,320),anti_aliasing=True))
            Validation_sample['image'] = Validation_sample['image'].permute(0,2,1,3,4)
    
            param_values = [v for v in Validation_sample.values()]
            image,angle = param_values[0],param_values[3]
            image = image.to(device)
            prediction = network(image)
            prediction = prediction.reshape(parameters.batch_size,parameters.seq_len)
            labels = angle.float().reshape(parameters.batch_size,parameters.seq_len).to(device)
            del param_values, image, angle
            if labels.shape[0]!=prediction.shape[0]:
                prediction = prediction[-labels.shape[0],:]
            validation_loss_angle = criterion(prediction,labels)
            val_losses.update(validation_loss_angle.item())
    wandb.log({'training_loss': losses.avg, 'val_loss': val_losses.avg})
    