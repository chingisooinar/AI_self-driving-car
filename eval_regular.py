#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 13:05:49 2021

@author: chingis
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 12:24:40 2021

@author: chingis
"""
from PIL import Image
from itertools import product
from easydict import EasyDict as edict
from tqdm import tqdm
import torch.backends.cudnn as cudnn
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from model.TransferLearning import TLearning
from model.DAVE2 import DAVE2
from DataLoading import EvalDataset as ED
from DataLoading import ConsecutiveBatchSampler as CB
import json
import numpy as np

#import wandb
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
    batch_size = 64,
    num_workers = 8,
    image_size = (120, 320),
    model_name = 'dave2',
    checkpoint='saved_models/dave2/DAVE2.tar'
)


if parameters.model_name == 'dave2':
    model_object = DAVE2
elif parameters.model_name == 'transfer':
    model_object = TLearning
else:
    raise KeyError("Unknown Architecture")
    
network = model_object()  
network.load_state_dict(torch.load(parameters.checkpoint))
network.to(device)


validation_set = ED.EvalDataset(csv_file='/home/chingis/self-driving-car/CH2_final_evaluation.csv',
                             root_dir='/home/chingis/self-driving-car/center/',
                             transform=transforms.Compose([              
                                 transforms.ToTensor(),
                                 transforms.Normalize(0.5, 0.5)
                                 ]), 
                             public=False, 
                             private=False, 
                             img_size=parameters.image_size,
                             optical_flow=False)

validation_loader = DataLoader(validation_set,  shuffle=False, num_workers=parameters.num_workers, batch_size=parameters.batch_size)
criterion =  torch.nn.MSELoss()
criterion.to(device)

network.eval()
predictions = []
# Calculation on Validation Loss
val_losses = AverageMeter()
with torch.no_grad():    
    for Validation_sample in tqdm(validation_loader):
        param_values = [v for v in Validation_sample.values()]
        image, angle, idx = param_values
        cur_bn = image.shape[0]
        image = image.to(device)

        prediction = network(image)
        prediction = prediction.reshape(-1,1)
        labels = angle.float().reshape(-1,1).to(device)

        validation_loss_angle = torch.sqrt(criterion(prediction,labels)+ 1e-6)
        val_losses.update(validation_loss_angle.item())

        predictions.extend(list(zip(idx.detach().cpu().flatten().numpy().tolist(), prediction.detach().cpu().flatten().numpy().tolist() )))


print(val_losses.avg)

res_dic = {} 
for pair in predictions:
    idx, angle = pair
    if idx not in res_dic.keys():
        res_dic[idx] = angle


a_file = open("predictions.json", "w")
a_file = json.dump(res_dic, a_file)