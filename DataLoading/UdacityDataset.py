#!/usr/bin/env python
# coding: utf-8




import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np

# defining customized Dataset class for Udacity
from .aug_utils import translate_image, add_random_shadow, change_image_brightness_rgb, do_rotate, blur
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms, utils
import random



class UdacityDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, select_camera=None, slice_frames=None, select_ratio=1.0, select_range=None, optical_flow=True):
        
        assert select_ratio >= -1.0 and select_ratio <= 1.0 # positive: select to ratio from beginning, negative: select to ration counting from the end
        
        camera_csv = pd.read_csv(csv_file)
        if select_camera:
            assert select_camera in ['left_camera', 'right_camera', 'center_camera'], "Invalid camera: {}".format(select_camera)
            camera_csv = camera_csv[camera_csv['frame_id']==select_camera]
        
        csv_len = len(camera_csv)
        if slice_frames:
            csv_selected = camera_csv[0:0] # empty dataframe
            for start_idx in range(0, csv_len, slice_frames):
                if select_ratio > 0:
                    end_idx = int(start_idx + slice_frames * select_ratio)
                else:
                    start_idx, end_idx = int(start_idx + slice_frames * (1 + select_ratio)), start_idx + slice_frames

                if end_idx > csv_len:
                    end_idx = csv_len
                if start_idx > csv_len:
                    start_idx = csv_len
                csv_selected = csv_selected.append(camera_csv[start_idx:end_idx])
            self.camera_csv = csv_selected
        elif select_range:
            csv_selected = camera_csv.iloc[select_range[0]: select_range[1]]
            self.camera_csv = csv_selected
        else:
            self.camera_csv = camera_csv
            
        self.root_dir = root_dir
        self.transform = transform
        self.optical_flow = optical_flow
        # Keep track of mean and cov value in each channel
        self.mean = {}
        self.std = {}
        for key in ['angle', 'torque', 'speed']:
            self.mean[key] = np.mean(camera_csv[key])
            self.std[key] = np.std(camera_csv[key])
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=0.5, std=0.5)
    def __len__(self):
        return len(self.camera_csv)
    
    def read_data_single(self, idx, flip_horizontally, translate, rotate):
        path = os.path.join(self.root_dir, self.camera_csv['filename'].iloc[idx])
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image[65:-25,:,:]
        # angle independent augs
        if random.uniform(0, 1) > 0.5:
            image = change_image_brightness_rgb(image)
        if random.uniform(0, 1) > 0.5:
            image = add_random_shadow(image)
        if random.uniform(0, 1) > 0.5:
            image = blur(image)

        angle = self.camera_csv['angle'].iloc[idx]
        # angle modifying augs
        if rotate is not None:
            image, angle = do_rotate(image, angle, *rotate)

        if translate is not None:
            image, angle  = translate_image(image, angle, *translate)
    
        angle_t = torch.tensor(angle)

  
        if self.transform:
            image_transformed = self.transform(image)

        if self.optical_flow:
            if idx != 0:
                path = os.path.join(self.root_dir, self.camera_csv['filename'].iloc[idx - 1])
                prev = cv2.imread(path)
                prev = cv2.cvtColor(prev, cv2.COLOR_BGR2RGB)
                prev = prev[65:-25,:,:]
                prev = cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY)
            else:
                prev = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            cur = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            # Use Hue, Saturation, Value colour model
            flow = cv2.calcOpticalFlowFarneback(prev, cur, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            hsv = np.zeros(image.shape, dtype=np.uint8)
            hsv[..., 1] = 255
            
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            optical_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            optical_rgb = self.transform(optical_rgb)
            del image
            if flip_horizontally:
                image_transformed = torch.fliplr(image_transformed)
                angle_t = angle_t * -1.0
                optical_rgb = torch.fliplr(optical_rgb)

            return image_transformed, angle_t, optical_rgb
        
        if self.transform:
            del image
            image = image_transformed
        if flip_horizontally:
            image = torch.fliplr(image)
            angle_t = angle_t * -1.0
    
        return image, angle_t
    
    def read_data(self, idx, flip_horizontally, translate, rotate):
        if isinstance(idx, list):
            data = None
            for i in idx:
                new_data = self.read_data(i, flip_horizontally, translate, rotate)
                if data is None:
                    data = [[] for _ in range(len(new_data))]
                for i, d in enumerate(new_data):
                    data[i].append(new_data[i])
                del new_data
            if self.optical_flow:
                for stack_idx in [0, 1, 2]: # we don't stack timestamp and frame_id since those are string data
                    data[stack_idx] = torch.stack(data[stack_idx])
            else:
                for stack_idx in [0, 1]: # we don't stack timestamp and frame_id since those are string data
                    data[stack_idx] = torch.stack(data[stack_idx])
            
            return data
        
        else:
            return self.read_data_single(idx, flip_horizontally, translate, rotate)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        flip_horizontally = random.uniform(0, 1) > 0.5
        translate = None
        if random.uniform(0, 1) > 0.5: # translate
            translation_x = np.random.randint(-60, 61) 
            translation_y = np.random.randint(-20, 21)
            translate = (translation_x, translation_y, 0.35/100.0)
        rotate = None
        if random.uniform(0, 1) > 0.5: # rotate
            random_rot = np.random.randint(5, 15+1)
            rotation_angle = random.choice([-random_rot, random_rot])
            rotate = (rotation_angle,)

        data = self.read_data(idx, flip_horizontally, translate, rotate)

        sample = {'image': data[0],
                  'angle': data[1]}
        if self.optical_flow:
            sample['optical'] = data[2]
        
        del data
        
        return sample

