#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 21:15:21 2025

@author: guillaume
"""

import torch as t
import math 

def FC_forward(X,W,b):
    Z = X.matmul(W) + b #NxH
    return Z

def FC_backward(dc_dZ, X, W, b):
    dc_dX = dc_dZ.matmul(W.T)
    dc_dW = X.T.matmul(dc_dZ)
    dc_db = dc_dZ.sum(axis=0)
    return dc_dX, dc_dW, dc_db

def relu_forward(X):
    Z = t.maximum(t.tensor(0.),X)
    return Z

def relu_backward(dc_dZ, X):
    dc_dX = dc_dZ.clone()
    dc_dX[X<0.] = 0.
    return dc_dX
    
class MLP:
    def __init__(self, H, C, D):

        self.C = C #output size i.e number of classes for a classification task
        self.D = D #input size (784 for MNIST)
        self.H = H #hidden layer size
        

        #parameters
        self.W1 = (math.sqrt(6./self.D))*(2*(t.rand(size=(self.D,self.H))-0.5))
        self.b1 = (1./math.sqrt(self.D))*(2*(t.rand(size=(self.H,))-0.5))
        self.W3 = (math.sqrt(6./self.H))*(2*(t.rand(size=(self.H,self.C))-0.5))
        self.b3 = (1./math.sqrt(self.H))*(2*(t.rand(size=(self.C,))-0.5))
        
        #gradients
        self.dc_dW1 = t.zeros_like(self.W1)
        self.dc_db1 = t.zeros_like(self.b1)
        self.dc_dW3 = t.zeros_like(self.W3)
        self.dc_db3 = t.zeros_like(self.b3)
        

        
    def forward(self,X):
    
        X1 = FC_forward(X, self.W1, self.b1) #NxH
        X2 = relu_forward(X1) #NxH
        S = FC_forward(X2, self.W3, self.b3) #NxC
    
        return X,X1,X2,S
    
    def backward(self,dc_dS, S, X2, X1, X0):
        
        dc_dX2, dc_dW3, dc_db3 = FC_backward(dc_dS, X2, self.W3, self.b3)
        self.dc_dW3 += dc_dW3
        self.dc_db3 += dc_db3
        
        dc_dX1 = relu_backward(dc_dX2, X1)
        
        dc_dX0, dc_dW1, dc_db1 = FC_backward(dc_dX1, X0, self.W1, self.b1)
        self.dc_dW1 += dc_dW1
        self.dc_db1 += dc_db1
        
        
        return
        

    
class GradientDescentWithMomentum:
    def __init__(self, model, beta, lr):
        
        self.model = model
        self.beta= beta
        self.lr = lr
        
        #momentum
        self.VW1 = t.zeros_like(self.model.W1)
        self.Vb1 = t.zeros_like(self.model.b1)
        self.VW3 = t.zeros_like(self.model.W3)
        self.Vb3 = t.zeros_like(self.model.b3)
        
        self.is_init = True
    def step(self):
        with t.no_grad():
            if(self.is_init == True):
                self.VW1 = self.model.dc_dW1      
                self.VW3 = self.model.dc_dW3         
                self.Vb1 = self.model.dc_db1           
                self.Vb3 = self.model.dc_db3
                
                self.is_init = False
            else:            
                self.VW1 = self.beta*self.VW1 + self.model.dc_dW1
                self.VW3 = self.beta*self.VW3 + self.model.dc_dW3
                self.Vb1 = self.beta*self.Vb1 + self.model.dc_db1
                self.Vb3 = self.beta*self.Vb3 + self.model.dc_db3
            
            self.model.W1 -= self.lr*self.VW1
            self.model.W3 -= self.lr*self.VW3
            self.model.b1 -= self.lr*self.Vb1
            self.model.b3 -= self.lr*self.Vb3
    
    def zero_grad(self):
        self.model.dc_dW1.zero_()
        self.model.dc_db1.zero_()
        self.model.dc_dW3.zero_()
        self.model.dc_db3.zero_()
        
    
def logsoftmax(x):
    x_shift = x - t.amax(x, axis=1, keepdims=True)
    return x_shift - t.log(t.exp(x_shift).sum(axis=1, keepdims=True))   
    
def softmax(x):
    e_x = t.exp(x - t.amax(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)
    
def crossEntropyLoss(S, y):
    N = y.shape[0]
    P = softmax(S.to(t.float64))
    log_p = logsoftmax(S.to(t.float64))
    a = log_p[t.arange(N),y]
    l = -a.sum()/N
    dc_dS = P
    dc_dS[t.arange(N),y] -= 1
    dc_dS = dc_dS/N
    return (l.to(t.float32), dc_dS.to(t.float32))