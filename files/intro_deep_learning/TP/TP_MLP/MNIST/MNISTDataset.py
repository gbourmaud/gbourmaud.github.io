#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 11:02:53 2022

@author: guillaume
"""
import torch as t
import PIL.Image as Image
import torchvision.transforms as T
import os


class MNISTDataset(t.utils.data.Dataset):
    def __init__(self, MNIST_dir, mean_norm=0., std_norm=1.):
        
        self.mean_norm = mean_norm
        self.std_norm = std_norm
        
        self.MNIST_dir = MNIST_dir
        self.num_classes = 10
        
        self.img_list = []
        self.label_list = []
        for i in range(self.num_classes):
            path_cur = os.path.join(self.MNIST_dir,'{}'.format(i))
            img_list_cur = os.listdir(path_cur)
            
            img_list_cur = [os.path.join('{}'.format(i), file) for file in img_list_cur]

            self.img_list += img_list_cur
            
            label_list_cur = [i] * len(img_list_cur)
            self.label_list += label_list_cur
            
    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, idx):
        
        img_path = os.path.join(self.MNIST_dir, self.img_list[idx])
        
        I_PIL = Image.open(img_path) #load an image as PIL
        
        I = T.ToTensor()(I_PIL) #PIL to torch.Tensor
        
        #NORMALIZATION
        I -= self.mean_norm
        I /= self.std_norm
    
        return I, t.tensor(self.label_list[idx]), img_path
                
            
