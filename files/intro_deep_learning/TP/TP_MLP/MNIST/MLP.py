import torch as t
import math 
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, H):
        super(MLP, self).__init__()
        
        self.C = 10
        self.D = 784
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
    
        return S