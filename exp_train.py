#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 12:24:40 2021

@author: chingis
"""

from easydict import EasyDict as edict
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
from DataLoading import UdacityDataset as UD
from DataLoading import ConsecutiveBatchSampler as CB
from model.MotionTransformer import MotionTransformer
from model.SimpleTransformer import SimpleTransformer
from model.LSTM import SequenceModel
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
    learning_rate = 0.0001,
    batch_size = 13,
    seq_len = 5,
    num_workers = 8,
    model_name = 'SimpleTransformer',
    normalization = ([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225]),
    image_size=(224, 224),
    epochs=161,
    all_frames=False,
    optical_flow=False
)
if parameters.model_name == 'LSTM':
    model_object = SequenceModel
elif parameters.model_name == 'MotionTransformer' :
    model_object = MotionTransformer
elif parameters.model_name == 'SimpleTransformer' :
    model_object = SimpleTransformer
else:
    raise KeyError("Unknown Architecture")
    
network = model_object(parameters.seq_len) 
network.to(device)
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
                                 #transforms.Resize((224,224)),#(120,320)
                                 transforms.ToTensor(),
                                 transforms.Normalize(*parameters.normalization)
                                 ]),
                             img_size=parameters.image_size,
                             seq_len=parameters.seq_len,
                             optical_flow=parameters.optical_flow,
                             select_camera='center_camera',
                             select_range=(0,split_point))
validation_set = UD.UdacityDataset(csv_file='/home/chingis/self-driving-car/output/interpolated.csv',
                             root_dir='/home/chingis/self-driving-car/output/',
                             transform=transforms.Compose([
                                # transforms.Resize((224,224)),
                                 transforms.ToTensor(),
                                 transforms.Normalize(*parameters.normalization)
                                 ]),
                             img_size=parameters.image_size,
                             seq_len=parameters.seq_len,
                             optical_flow=parameters.optical_flow,
                             select_camera='center_camera',
                             select_range=(split_point,dataset_size))


training_cbs = CB.ConsecutiveBatchSampler(data_source=training_set, batch_size=parameters.batch_size,use_all_frames=parameters.all_frames, shuffle=True, drop_last=False, seq_len=parameters.seq_len)
training_loader = DataLoader(training_set, sampler=training_cbs, num_workers=parameters.num_workers, collate_fn=(lambda x: x[0]))

validation_cbs = CB.ConsecutiveBatchSampler(data_source=validation_set, batch_size=parameters.batch_size, use_all_frames=False, shuffle=False, drop_last=False, seq_len=parameters.seq_len)
validation_loader = DataLoader(validation_set, sampler=validation_cbs, num_workers=parameters.num_workers, collate_fn=(lambda x: x[0]))
criterion = torch.nn.MSELoss()
criterion.to(device)
speed_criterion =  torch.nn.SmoothL1Loss()
speed_criterion.to(device)

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = parameters.learning_rate
    if epoch in [30, 90, 150]:
        lr = parameters.learning_rate * 0.1
        parameters.learning_rate = lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

for epoch in range(parameters.epochs):
    train_angle_losses = AverageMeter()
    train_speed_losses = AverageMeter()
    torch.save(network.state_dict(), "saved_models/transformer/epoch.tar")
    network.train()
    adjust_learning_rate(optimizer, epoch)
    # Calculation on Training Loss
    for training_sample in tqdm(training_loader):
        param_values = [v for v in training_sample.values()]
        if parameters.optical_flow:
            image, angle, optical, speed = param_values
            optical = optical.to(device)
            speed = speed.float().reshape(-1, 1).to(device)
        else:
            image, angle = param_values
        cur_bn = image.shape[0]
        #image = image.permute(0,2,1,3,4)
        #optical = optical.permute(0,2,1,3,4)

        loss = 0
        image = image.to(device)
        #optical = optical.to(device)
        #with torch.cuda.amp.autocast():
        if parameters.optical_flow:
            angle_hat, speed_hat = network(image, optical)  
            speed_hat = speed_hat.reshape(-1, 1)
            training_loss_speed = speed_criterion(speed_hat,speed)
            loss += training_loss_speed
            train_speed_losses.update(training_loss_speed.item())
        else:
            angle_hat = network(image)
        #print(prediction.shape, angle.squeeze()[: , -angle.shape[1]:].shape)
        angle_hat = angle_hat.reshape(-1, 1)
        angle = angle.float().reshape(-1, 1).to(device)
        training_loss_angle = torch.sqrt(criterion(angle_hat,angle) + 1e-6)
        loss += training_loss_angle

        train_angle_losses.update(training_loss_angle.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    
    print("Done")
    report = {}
    if split_point != dataset_size:
        network.eval()
        val_speed_losses = AverageMeter()
        val_angle_losses = AverageMeter()
        with torch.no_grad():    
            for Validation_sample in tqdm(validation_loader):
                param_values = [v for v in Validation_sample.values()]
                if parameters.optical_flow:
                    image, angle, optical, speed = param_values
                    optical = optical.to(device)
                    speed = speed.float().reshape(-1, 1).to(device)
                else:
                    image, angle = param_values
                cur_bn = image.shape[0]
                #image = image.permute(0,2,1,3,4)
                #optical = optical.permute(0,2,1,3,4)
                loss = 0
                image = image.to(device)
                #optical = optical.to(device)
                if parameters.optical_flow:
                    angle_hat, speed_hat = network(image, optical)  
                    speed_hat = speed_hat.reshape(-1, 1)
                    validation_loss_speed = speed_criterion(speed_hat,speed)
                    loss += validation_loss_speed
                    val_speed_losses.update(validation_loss_speed.item())
                else:
                    angle_hat = network(image)
                angle_hat = angle_hat.reshape(-1, 1)
                angle = angle.float().reshape(-1, 1).to(device)
    
                validation_loss_angle = torch.sqrt(criterion(angle_hat,angle) + 1e-6)
                loss += validation_loss_angle
    
                val_angle_losses.update(validation_loss_angle.item())
        report['val_loss'] = val_angle_losses.avg
    report['training_loss'] = train_angle_losses.avg
    if parameters.optical_flow:
        report['training_speed_loss'] = train_speed_losses.avg
        if split_point != dataset_size:
            report['validation_speed_loss'] = val_speed_losses.avg    

    wandb.log(report)
    