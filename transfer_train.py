#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 12:54:17 2021

@author: chingis
"""
from PIL import Image
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


network = TLearning()
network = torch.nn.DataParallel(network).to(device)

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
                             optical_flow=False,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize(0.5, 0.5)
                                 ]),
                             select_camera='center_camera',
                             select_range=(0,split_point))
validation_set = UD.UdacityDataset(csv_file='/home/chingis/self-driving-car/output/interpolated.csv',
                             root_dir='/home/chingis/self-driving-car/output/',
                             optical_flow=False,
                             transform=transforms.Compose([transforms.ToTensor(),
                                                           transforms.Normalize(0.5, 0.5)]),
                             select_camera='center_camera',
                             select_range=(split_point,dataset_size))

training_loader = DataLoader(training_set, shuffle=True, num_workers=parameters.num_workers, batch_size=parameters.batch_size)
validation_loader = DataLoader(validation_set,  shuffle=False, num_workers=parameters.num_workers, batch_size=parameters.batch_size)
criterion =  torch.nn.MSELoss()
criterion.to(device)
for epoch in range(32):
    losses = AverageMeter()
    torch.save(network.state_dict(), "saved_models/TLearning/epoch-{}.tar".format(epoch))
    network.train()
    # Calculation on Training Loss
    for training_sample in tqdm(training_loader):
        training_sample['image'] = training_sample['image'].squeeze()
        training_sample['image'] = torch.Tensor(resize(training_sample['image'], (parameters.batch_size,3,224,224),anti_aliasing=True))

        
        param_values = [v for v in training_sample.values()]
        image, angle = param_values[0], param_values[3]


        image = image.to(device)
        prediction = network(image)

        prediction = prediction.reshape(parameters.batch_size,1)
        labels = angle.float().reshape(parameters.batch_size,1).to(device)

        if labels.shape[0]!=prediction.shape[0]:
            prediction = prediction[-labels.shape[0],:]

        training_loss_angle =criterion(prediction,labels)

        losses.update(training_loss_angle.item())
        optimizer.zero_grad()# zero the gradient that are being held in the Grad attribute of the weights
        training_loss_angle.backward() # calculate the gradients
        optimizer.step() # finishing calculation on gradient
    

    network.eval()
    # Calculation on Validation Loss
    val_losses = AverageMeter()
    with torch.no_grad():    
        for Validation_sample in tqdm(validation_loader):
            Validation_sample['image'] = Validation_sample['image'].squeeze()
            Validation_sample['image'] = torch.Tensor(resize(Validation_sample['image'], (parameters.batch_size,3,224,224),anti_aliasing=True))


            param_values = [v for v in training_sample.values()]
            image, angle = param_values[0], param_values[3]
            image = image.to(device)
            prediction = network(image)
            prediction = prediction.reshape(parameters.batch_size,1)
            labels = angle.float().reshape(parameters.batch_size,1).to(device)
            del param_values, image, angle
            if labels.shape[0]!=prediction.shape[0]:
                prediction = prediction[-labels.shape[0],:]
            validation_loss_angle = criterion(prediction,labels)
            val_losses.update(validation_loss_angle.item())
    wandb.log({'training_loss': losses.avg, 'val_loss': val_losses.avg})
    