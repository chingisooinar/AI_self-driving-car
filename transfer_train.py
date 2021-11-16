#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 12:54:17 2021

@author: chingis
"""
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
import random
from model.TransferLearning import TLearning
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb
from DataLoading.aug_utils import change_image_brightness_rgb, add_random_shadow
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
    learning_rate = 0.001,
    batch_size = 32,
    seq_len = 5,
    num_workers = 8,
)


network = TLearning().to(device)
#network = torch.nn.DataParallel(network).to(device)

wandb.init(config=parameters, project='self-driving-car')
wandb.watch(network)
optimizer = optim.Adam(network.parameters(),lr = parameters.learning_rate,betas=(0.9, 0.999), eps=1e-08)

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
                                 transforms.Resize((120,320)),
                                 transforms.ToTensor(),
                                 transforms.Normalize(0.5, 0.5)
                                 ]),
                             optical_flow=False,
                             select_camera='center_camera',
                             select_range=(0,split_point))
validation_set = UD.UdacityDataset(csv_file='/home/chingis/self-driving-car/output/interpolated.csv',
                             root_dir='/home/chingis/self-driving-car/output/',
                             transform=transforms.Compose([
                                 transforms.Resize((120,320)),
                                 transforms.ToTensor(),
                                 transforms.Normalize(0.5, 0.5)]),
                             optical_flow=False,
                             select_camera='center_camera',
                             select_range=(split_point,dataset_size))

training_loader = DataLoader(training_set, shuffle=True, num_workers=parameters.num_workers, batch_size=parameters.batch_size)
validation_loader = DataLoader(validation_set,  shuffle=False, num_workers=parameters.num_workers, batch_size=parameters.batch_size)
criterion =  torch.nn.MSELoss()
criterion.to(device)
scaler = torch.cuda.amp.GradScaler()

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = parameters.learning_rate
    if epoch in [30, 90, 150]:
        lr = parameters.learning_rate * 0.1
        parameters.learning_rate = lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

for epoch in range(200):
    losses = AverageMeter()
    torch.save(network.state_dict(), "saved_models/TLearning/epoch-{}.tar".format(epoch))
    network.train()
    adjust_learning_rate(optimizer, epoch)
    # Calculation on Training Loss
    for training_sample in tqdm(training_loader):
        param_values = [v for v in training_sample.values()]
        image, angle = param_values
        cur_bn = image.shape[0]

        
        image = image.to(device)

        prediction = network(image)
        prediction = prediction.reshape(-1, 1)
        labels = angle.float().reshape(-1, 1).to(device)
        training_loss_angle = torch.sqrt(criterion(prediction,labels) + 1e-6)
        losses.update(training_loss_angle.item())
        optimizer.zero_grad()
        training_loss_angle.backward()
        optimizer.step()


    
    print("Done")
    network.eval()
    # Calculation on Validation Loss
    val_losses = AverageMeter()
    with torch.no_grad():    
        for Validation_sample in tqdm(validation_loader):
            param_values = [v for v in Validation_sample.values()]
            image, angle = param_values
            cur_bn = image.shape[0]
            image = image.to(device)

            prediction = network(image)
            prediction = prediction.reshape(-1,1)
            labels = angle.float().reshape(-1,1).to(device)

            validation_loss_angle = torch.sqrt(criterion(prediction,labels)+ 1e-6)
            val_losses.update(validation_loss_angle.item())
    wandb.log({'training_loss': losses.avg, 'val_loss': val_losses.avg})
    
    
