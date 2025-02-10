
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
import torch.nn as nn

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

    
class MLP(nn.Module):
    def __init__(self, H):
        super(MLP, self).__init__()
        
        self.C = 3
        self.D = 2
        self.H = H
        
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

optimizer = t.optim.SGD(model.parameters(), lr=lr, momentum=beta)  
loss = nn.CrossEntropyLoss()
it = 0
while 1:    
    #Forward Pass
    X0,X1,X2,S = model.forward(X)
    
    #Compute Loss
    c = loss(S, y)
    
    #Print Loss and Classif Accuracy
    pred = t.argmax(S, axis=1)
    acc = (t.argmax(S, axis=1) == y).to(t.float32).sum()/N
    print('Iter {} | Training Loss = {} | Training Accuracy = {}%'.format(it,c,acc*100))

    #Backward Pass (Compute Gradient)
    optimizer.zero_grad()
    c.backward()
    
    #Update Parameters
    optimizer.step()
    it += 1
    
    c_seq.append(c.data)
    it_seq.append(it)
    if(it%10==0):
        #Plot decision boundary
        axs1[0].cla()
        for i in range(C):
            x_c = X[y==i,:].detach()
            axs1[0].plot(x_c[:,0],x_c[:,1],style_per_class[i],markersize=7, markeredgewidth=3.)
        utils.plot_contours(axs1[0], model, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
        
        line_loss.remove()
        line_loss, = axs1[1].plot(it_seq,c_seq,'r',label='Training loss')
        axs1[1].legend()
        fig1.canvas.draw()
        fig1.canvas.flush_events()
        plt.pause(0.1)



    
