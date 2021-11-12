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

from skimage.transform import resize
from itertools import product
from easydict import EasyDict as edict
from tqdm import tqdm
import torch.backends.cudnn as cudnn
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from DataLoading import EvalDataset as ED
from DataLoading import ConsecutiveBatchSampler as CB

from model import Convolution3D as CNN3D

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
    batch_size = 32,
    seq_len = 5,
    num_workers = 2,
)


network = CNN3D.Convolution3D().to(device)

network.load_state_dict(torch.load('saved_models/CNN3D/epoch-31.tar'))
#wandb.init(config=parameters, project='self-driving-car')
#wandb.watch(network)

validation_set = ED.EvalDataset(csv_file='/home/chingis/self-driving-car/CH2_final_evaluation.csv',
                             root_dir='/home/chingis/self-driving-car/center/',
                             transform=transforms.Compose([transforms.ToTensor()]))


validation_cbs = CB.ConsecutiveBatchSampler(data_source=validation_set, batch_size=parameters.batch_size, shuffle=False, drop_last=False, seq_len=parameters.seq_len)
validation_loader = DataLoader(validation_set, sampler=validation_cbs, num_workers=parameters.num_workers, collate_fn=(lambda x: x[0]))
criterion =  torch.nn.MSELoss()
criterion.to(device)

losses = AverageMeter()
print("Done")
network.eval()
# Calculation on Validation Loss
val_losses = AverageMeter()
with torch.no_grad():    
    for Validation_sample in tqdm(validation_loader):
        Validation_sample['image'] = torch.Tensor(resize(Validation_sample['image'], (parameters.batch_size,parameters.seq_len,3,120,320),anti_aliasing=True))
        Validation_sample['image'] = Validation_sample['image'].permute(0,2,1,3,4)

        param_values = [v for v in Validation_sample.values()]
        image,angle = param_values[0],param_values[1]
        image = image.to(device)
        prediction = network(image)
        prediction = prediction.reshape(parameters.batch_size,parameters.seq_len)
        labels = angle.float().reshape(parameters.batch_size,parameters.seq_len).to(device)
        del param_values, image, angle
        if labels.shape[0]!=prediction.shape[0]:
            prediction = prediction[-labels.shape[0],:]
        validation_loss_angle = criterion(prediction,labels)
        val_losses.update(validation_loss_angle.item())
#wandb.log({'training_loss': losses.avg, 'val_loss': val_losses.avg})
print(val_losses.avg)