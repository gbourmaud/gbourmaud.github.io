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
    
        return X,X1,X2,S

    
class GradientDescentWithMomentum:
    def __init__(self, model, beta, lr):
        
        self.it = 0
        
        self.model = model
        self.beta= beta
        self.lr = lr
        
        #momentum
        self.VW1 = t.zeros_like(self.model.fc1.weight)
        self.Vb1 = t.zeros_like(self.model.fc1.bias)
        self.VW3 = t.zeros_like(self.model.fc3.weight)
        self.Vb3 = t.zeros_like(self.model.fc3.bias)
        
        self.is_init = True
    def step(self):
        with t.no_grad():
            if(self.is_init == True):
                self.VW1 = self.model.fc1.weight.grad        
                self.VW3 = self.model.fc3.weight.grad            
                self.Vb1 = self.model.fc1.bias.grad            
                self.Vb3 = self.model.fc3.bias.grad
                
                self.is_init = False
            else:            
                self.VW1 = self.beta*self.VW1 + self.model.fc1.weight.grad
                self.VW3 = self.beta*self.VW3 + self.model.fc3.weight.grad
                self.Vb1 = self.beta*self.Vb1 + self.model.fc1.bias.grad
                self.Vb3 = self.beta*self.Vb3 + self.model.fc3.bias.grad
            
            self.model.fc1.weight -= self.lr*self.VW1
            self.model.fc3.weight -= self.lr*self.VW3
            self.model.fc1.bias -= self.lr*self.Vb1
            self.model.fc3.bias -= self.lr*self.Vb3

    
    def zero_grad(self):
        self.model.fc1.weight.grad = None
        self.model.fc1.bias.grad = None
        self.model.fc3.weight.grad = None
        self.model.fc3.bias.grad = None
        

    
