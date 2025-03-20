import PIL.Image as Image
from os import path
from os import listdir
import numpy as np
import math

def load_MNIST(path_MNIST_bmp, mean_norm=0., std_norm=1.):
    
    filenames = listdir(path_MNIST_bmp)
    
    imgs_train = np.zeros((60000,28,28))
    labels_train = np.zeros(60000,dtype=np.uint8)
    n_train = 0
    
    imgs_test = np.zeros((10000,28,28))
    labels_test = np.zeros(10000,dtype=np.uint8)
    n_test = 0
    
    W=H=28
    for i, filename in enumerate(filenames):
        
        f_name, f_ext = path.splitext(filename)
        set_type, class_type ,num_im = f_name.split('_')
        
        num_im = int(num_im)
        img = np.array(Image.open(path.join(path_MNIST_bmp,filename)))/255.
        
        N_W = math.ceil(math.sqrt(num_im))
        N_H = math.ceil(num_im/N_W)
        
        im_array_ext = img.reshape(N_H,H,W*N_W).transpose((1,0,2))
        im_array_ext = im_array_ext.reshape(H,N_W*N_H,W).transpose((1,0,2))

        im_array = im_array_ext[:num_im,:,:]        
        
        if(set_type=='train'):
            imgs_train[n_train:n_train+num_im,:,:] = (im_array-mean_norm)/std_norm
            labels_train[n_train:n_train+num_im] = int(class_type)
            n_train += num_im
        elif(set_type=='test'):
            imgs_test[n_test:n_test+num_im,:,:] = (im_array-mean_norm)/std_norm
            labels_test[n_test:n_test+num_im] = int(class_type)
            n_test += num_im
                
    assert n_test == 10000
    assert n_train == 60000
        
        
    
    return imgs_train, labels_train, imgs_test, labels_test
    
def FC_forward(X,W,b):
    Z = X.dot(W) + b #NxH
    return Z

def FC_backward(dc_dZ, X, W, b):
    dc_dX = dc_dZ.dot(W.T)
    dc_dW = X.T.dot(dc_dZ)
    dc_db = dc_dZ.sum(axis=0)
    return dc_dX, dc_dW, dc_db

def relu_forward(X):
    Z = np.maximum(0.,X)
    return Z

def relu_backward(dc_dZ, X):
    dc_dX = dc_dZ.copy()
    dc_dX[X<0.] = 0.
    return dc_dX
    
    
class MLP:
    def __init__(self, H, C, D):

        self.C = C #output size i.e number of classes for a classification task
        self.D = D #input size (784 for MNIST)
        self.H = H #hidden layer size
        

        #parameters
        self.W1 = (np.sqrt(6./self.D))*(2*(np.random.uniform(size=(self.D,self.H))-0.5))
        self.b1 = (1./np.sqrt(self.D))*(2*(np.random.uniform(size=(self.H))-0.5))
        self.W3 = (np.sqrt(6./self.H))*(2*(np.random.uniform(size=(self.H,self.C))-0.5))
        self.b3 = (1./np.sqrt(self.H))*(2*(np.random.uniform(size=(self.C))-0.5))
        
        #gradients
        self.dc_dW1 = np.zeros_like(self.W1)
        self.dc_db1 = np.zeros_like(self.b1)
        self.dc_dW3 = np.zeros_like(self.W3)
        self.dc_db3 = np.zeros_like(self.b3)
        

        
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
        self.VW1 = np.zeros_like(self.model.W1)
        self.Vb1 = np.zeros_like(self.model.b1)
        self.VW3 = np.zeros_like(self.model.W3)
        self.Vb3 = np.zeros_like(self.model.b3)
        
        self.first_step = True
        
    def step(self):
        if(self.first_step):
            self.VW1 = self.model.dc_dW1      
            self.VW3 = self.model.dc_dW3         
            self.Vb1 = self.model.dc_db1           
            self.Vb3 = self.model.dc_db3
            self.first_step = False
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
        self.model.dc_dW1.fill(0.)
        self.model.dc_db1.fill(0.)
        self.model.dc_dW3.fill(0.)
        self.model.dc_db3.fill(0.)
        

        
def logsoftmax(x):
    x_shift = x - np.amax(x, axis=1, keepdims=True)
    return x_shift - np.log(np.exp(x_shift).sum(axis=1, keepdims=True))   
    
def softmax(x):
    e_x = np.exp(x - np.amax(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)
    
def crossEntropyLoss(S, y):
    N = y.shape[0]
    P = softmax(S.astype('double'))
    log_p = logsoftmax(S.astype('double'))
    a = log_p[np.arange(N),y]
    l = -a.sum()/N
    dc_dS = P
    dc_dS[np.arange(N),y] -= 1
    dc_dS = dc_dS/N
    return (l, dc_dS)