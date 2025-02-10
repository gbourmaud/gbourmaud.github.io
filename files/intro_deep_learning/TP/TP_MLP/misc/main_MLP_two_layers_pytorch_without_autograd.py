
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 11:22:25 2019

@author: guillaume
"""
import matplotlib.pyplot as plt
import utils
import torch as t
import math 

t.manual_seed(0)

#%% DEFINE AND PLOT DATA
    
style_per_class = ['xb', 'or', 'sg']
X = t.tensor([[1.2, 2.3, -0.7, 3.2, -1.3],[-3.4, 2.8, 1.2, -0.4, -2.3]]).T
X -= X.mean() #centering data (globally)
X /= X.std() #reduce data (globally)
y = t.tensor([0,0,1,1,2])

C = len(style_per_class)
N = X.shape[0]
xx, yy = utils.make_meshgrid(X[:,0], X[:,1], h=0.05)


fig1, axs1 = plt.subplots(ncols=2)
axs1[0].set_xlim(xx.min(), xx.max())
axs1[0].set_ylim(yy.min(), yy.max())
axs1[0].grid(True)

for i in range(C):
    x_c = X[y==i,:]
    axs1[0].plot(x_c[:,0],x_c[:,1],style_per_class[i],markersize=7, markeredgewidth=3.)

plt.pause(0.1)
   
        
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
    def __init__(self, H):

        self.C = 3
        self.D = 2
        self.H = H
        

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
        
    def step(self):
        with t.no_grad():
            if(i==0):
                self.VW1 = self.model.dc_dW1      
                self.VW3 = self.model.dc_dW3         
                self.Vb1 = self.model.dc_db1           
                self.Vb3 = self.model.dc_db3
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
        

#%% FIGURES
c_seq = []
it_seq = []
line_loss, = axs1[1].plot(it_seq,c_seq)
axs1[1].legend()
axs1[1].set_xlabel('Iterations')


#%% HYPERPARAMETERS
H = 300
lr = 1e-2 #learning rate
beta = 0.9 #momentum parameter

model = MLP(H)

optimizer = GradientDescentWithMomentum(model, beta, lr)

it = 0
while 1:    
    #Forward Pass
    X0,X1,X2,S = model.forward(X)
    
    #Compute Loss
    [c, dc_dS] = crossEntropyLoss(S, y)
    
    #Print Loss and Classif Accuracy
    pred = t.argmax(S, axis=1)
    acc = (t.argmax(S, axis=1) == y).to(t.float32).sum()/N
    print('Iter {} | Training Loss = {} | Training Accuracy = {}%'.format(it,c,acc*100))

    #Backward Pass (Compute Gradient)
    optimizer.zero_grad()
    model.backward(dc_dS, S, X2, X1, X0)
    
    #Update Parameters
    optimizer.step()
    it += 1
    
    c_seq.append(c)
    it_seq.append(it)
    if(it%10==0):
        #Plot decision boundary
        axs1[0].cla()
        for i in range(C):
            x_c = X[y==i,:]
            axs1[0].plot(x_c[:,0],x_c[:,1],style_per_class[i],markersize=7, markeredgewidth=3.)
        utils.plot_contours(axs1[0], model, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
        
        line_loss.remove()
        line_loss, = axs1[1].plot(it_seq,c_seq,'r',label='Training loss')
        axs1[1].legend()
        fig1.canvas.draw()
        fig1.canvas.flush_events()
        plt.pause(0.1)



    
