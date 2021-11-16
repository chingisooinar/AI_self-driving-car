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
from model.TransMotion import Convolution3D
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
    file = '3DCNN_Paper', # used to mark specific files in case that we want to check them on tensorboard
    batch_size = 64,
    seq_len = 5,
    num_workers = 8,
)


network = Convolution3D()
network.load_state_dict(torch.load('saved_models/CNN3D/epoch-31.tar'))
network.to(device)
#wandb.init(config=parameters, project='self-driving-car')
#wandb.watch(network)

validation_set = ED.EvalDataset(csv_file='/home/chingis/self-driving-car/CH2_final_evaluation.csv',
                             root_dir='/home/chingis/self-driving-car/center/',
                             transform=transforms.Compose([              
                                 transforms.Lambda(lambda x: Image.fromarray(np.uint8(x)).convert('RGB')),
                                 transforms.Resize((120,320)),
                                 transforms.ToTensor(),
                                 transforms.Normalize(0.5, 0.5)]), public=False, private=False)


validation_cbs = CB.ConsecutiveBatchSampler(data_source=validation_set, batch_size=parameters.batch_size, shuffle=False, drop_last=False, seq_len=parameters.seq_len)
validation_loader = DataLoader(validation_set, sampler=validation_cbs, num_workers=parameters.num_workers, collate_fn=(lambda x: x[0]))
criterion =  torch.nn.MSELoss()
criterion.to(device)
scaler = torch.cuda.amp.GradScaler()

network.eval()
predictions = []
# Calculation on Validation Loss
val_losses = AverageMeter()
with torch.no_grad():    
    for Validation_sample in tqdm(validation_loader):
        param_values = [v for v in Validation_sample.values()]
        image, angle,idx, optical = param_values
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
        predictions.extend(list(zip(idx.detach().cpu().flatten().numpy().tolist(), prediction.detach().cpu().flatten().numpy().tolist() )))


print(val_losses.avg)

res_dic = {} 
for pair in predictions:
    idx, angle = pair
    if idx not in res_dic.keys():
        res_dic[idx] = angle


a_file = open("predictions.json", "w")
a_file = json.dump(res_dic, a_file)