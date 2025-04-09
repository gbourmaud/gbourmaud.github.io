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


class MNISTTranslationDataset(t.utils.data.Dataset):
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
        
        translation = t.tensor([1, 1]).random_(to=28)
        
        img_path = os.path.join(self.MNIST_dir, self.img_list[idx])
        
        I_PIL = Image.open(img_path)
        
        I = T.ToTensor()(I_PIL)
        
        I_trans = t.zeros((1,56,56))
        I_trans[:,translation[1]:translation[1]+28, translation[0]:translation[0]+28] = I
        
        #NORMALIZATION
        I_trans -= self.mean_norm
        I_trans /= self.std_norm

        return I_trans, t.tensor(self.label_list[idx]), img_path
                
            