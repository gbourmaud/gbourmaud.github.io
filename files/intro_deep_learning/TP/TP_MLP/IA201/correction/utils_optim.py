#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 21:15:21 2025

@author: guillaume
"""

import torch as t
import torch.nn as nn
import math 

class MLP(nn.Module):
    def __init__(self, H, C, D):
        super(MLP, self).__init__()
        
        self.C = C #output size i.e number of classes for a classification task
        self.D = D #input size (784 for MNIST)
        self.H = H #hidden layer size
        
        self.fc1 = nn.Linear(self.D, self.H) 
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(self.H, self.C)  
        
        #init parameters
        with t.no_grad():
            self.fc1.weight.uniform_(-math.sqrt(6./self.D), math.sqrt(6./self.D))
            self.fc1.bias.uniform_(-1./math.sqrt(self.D), 1./math.sqrt(self.D))
            self.fc3.weight.uniform_(-math.sqrt(6./self.H),math.sqrt(6./self.H))
            self.fc3.bias.uniform_(-1./math.sqrt(self.H),1./math.sqrt(self.H))
        
    def forward(self,X):
    
        X1 = self.fc1(X) #NxH
        X2 = self.relu(X1) #NxH
        S = self.fc3(X2) #NxC
    
        return S

    


    
