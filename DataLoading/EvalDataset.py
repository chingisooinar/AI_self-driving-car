#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 12:57:21 2021

@author: chingis
"""

#!/usr/bin/env python
# coding: utf-8




import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np

# defining customized Dataset class for Udacity

import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms, utils
import random



class EvalDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None  ):
        
       
        
        camera_csv = pd.read_csv(csv_file)

        
        csv_len = len(camera_csv)

        self.camera_csv = camera_csv[camera_csv['public'] == 1]
            
        self.root_dir = root_dir
        self.transform = transform
        

    
    def __len__(self):
        return len(self.camera_csv)
    
    def read_data_single(self, idx):
        #print(str(self.camera_csv['frame_id'].iloc[idx]) + '.jpg')
        path = os.path.join(self.root_dir, str(self.camera_csv['frame_id'].iloc[idx]) + '.jpg')
        #print(path)
        image = cv2.imread(path)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image[65:-25,:,:]

        angle = self.camera_csv['steering_angle'].iloc[idx]
        if self.transform:
            image_transformed = self.transform(image)
            del image
            image = image_transformed
        angle_t = torch.tensor(angle)

        return image, angle_t
    
    def read_data(self, idx):
        if isinstance(idx, list):
            data = None
            for i in idx:
                new_data = self.read_data(i)
                if data is None:
                    data = [[] for _ in range(len(new_data))]
                for i, d in enumerate(new_data):
                    data[i].append(new_data[i])
                del new_data
                
            for stack_idx in [0, 1]: # we don't stack timestamp and frame_id since those are string data
                data[stack_idx] = torch.stack(data[stack_idx])
            
            return data
        
        else:
            return self.read_data_single(idx)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        data = self.read_data(idx)
        
        sample = {'image': data[0],
                  'angle': data[1]}
        
        del data
        
        return sample

