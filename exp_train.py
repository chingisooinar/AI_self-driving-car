#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 12:24:40 2021

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
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from DataLoading import UdacityDataset as UD
from DataLoading import ConsecutiveBatchSampler as CB
import random
from model.TransMotion import Convolution3D
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
    batch_size = 64,
    seq_len = 5,
    num_workers = 8,
)


network = Convolution3D().to(device)
#network = torch.nn.DataParallel(network).to(device)

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
                                 transforms.Lambda(lambda x: Image.fromarray(np.uint8(x)).convert('RGB')),
                                 transforms.Resize((120,320)),
                                 transforms.ToTensor(),
                                 transforms.Normalize(0.5, 0.5)
                                 ]),
                             select_camera='center_camera',
                             select_range=(0,split_point))
validation_set = UD.UdacityDataset(csv_file='/home/chingis/self-driving-car/output/interpolated.csv',
                             root_dir='/home/chingis/self-driving-car/output/',
                             transform=transforms.Compose([
                                 transforms.Lambda(lambda x: Image.fromarray(np.uint8(x)).convert('RGB')),
                                 transforms.Resize((120,320)),
                                 transforms.ToTensor(),
                                 transforms.Normalize(0.5, 0.5)]),
                             select_camera='center_camera',
                             select_range=(split_point,dataset_size))


training_cbs = CB.ConsecutiveBatchSampler(data_source=training_set, batch_size=parameters.batch_size, shuffle=False, drop_last=False, seq_len=parameters.seq_len)
training_loader = DataLoader(training_set, sampler=training_cbs, num_workers=parameters.num_workers, collate_fn=(lambda x: x[0]))

validation_cbs = CB.ConsecutiveBatchSampler(data_source=validation_set, batch_size=parameters.batch_size, shuffle=False, drop_last=False, seq_len=parameters.seq_len)
validation_loader = DataLoader(validation_set, sampler=validation_cbs, num_workers=parameters.num_workers, collate_fn=(lambda x: x[0]))
criterion =  torch.nn.MSELoss()
criterion.to(device)
scaler = torch.cuda.amp.GradScaler()
for epoch in range(32):
    losses = AverageMeter()
    torch.save(network.state_dict(), "saved_models/CNN3D/epoch-{}.tar".format(epoch))
    network.train()
    # Calculation on Training Loss
    for training_sample in tqdm(training_loader):
        param_values = [v for v in training_sample.values()]
        image, angle, optical = param_values
        cur_bn = image.shape[0]
        image = image.permute(0,2,1,3,4)
        optical = optical.permute(0,2,1,3,4)

        
        image = image.to(device)
        optical = optical.to(device)
        with torch.cuda.amp.autocast():
            prediction = network(image, optical)
            prediction = prediction.reshape(-1, 1)
            labels = angle.float().reshape(-1, 1).to(device)
            training_loss_angle =criterion(prediction,labels)

        losses.update(training_loss_angle.item())
        optimizer.zero_grad()
        scaler.scale(training_loss_angle).backward()
        scaler.step(optimizer)
        scaler.update()

    
    print("Done")
    network.eval()
    # Calculation on Validation Loss
    val_losses = AverageMeter()
    with torch.no_grad():    
        for Validation_sample in tqdm(validation_loader):
            param_values = [v for v in Validation_sample.values()]
            image, angle, optical = param_values
            cur_bn = image.shape[0]
            image = image.permute(0,2,1,3,4)
            optical = optical.permute(0,2,1,3,4)
            image = image.to(device)
            optical = optical.to(device)
            with torch.cuda.amp.autocast():
                prediction = network(image, optical)
                prediction = prediction.reshape(-1,1)
                labels = angle.float().reshape(-1,1).to(device)

                validation_loss_angle = criterion(prediction,labels)
            val_losses.update(validation_loss_angle.item())
    wandb.log({'training_loss': losses.avg, 'val_loss': val_losses.avg})
    