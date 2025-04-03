import numpy as np
import matplotlib.pyplot as plt
import utils_pytorch as utils
import torch as t

path_MNIST_bmp = './MNIST_bmp'

imgs_train, labels_train, imgs_test, labels_test = utils.load_MNIST(path_MNIST_bmp, mean_norm=0.13066047627384297, std_norm=0.30810780385646214)
n_train = imgs_train.shape[0]
n_test = imgs_test.shape[0]

ids = np.random.permutation(n_train)
imgs_val = imgs_train[ids[:10000],:,:]
labels_val = labels_train[ids[:10000]]
n_val = 10000

imgs_train = imgs_train[ids[10000:],:,:]
labels_train = labels_train[ids[10000:]]
n_train = n_train - 10000

ids = np.random.permutation(n_train)

plt.figure()
for i in range(8):
    for j in range(4):
        plt.subplot(4,8,i+1 + j*8)
        plt.imshow(imgs_train[ids[i+j*8],:,:])
        plt.title(labels_train[ids[i+j*8]])
        plt.axis('off')

vec_train = imgs_train.reshape((n_train,-1)) #transforms 50000x28x28 images into vectors 50000x784
vec_val = imgs_val.reshape((n_val,-1))

#%% Compute mean and std
mean = np.mean(imgs_train)
std = np.std(imgs_train)
print('Mean MNIST {}, Std MNIST {}'.format(mean, std))

#%% Validation function
def validation(vec_val, labels_val, model):
    print('Start validation')
    # Test the model    
    outputs = model.forward(vec_val)
    predicted = np.argmax(outputs[3], 1)
    total = len(labels_val)
    correct = (predicted == labels_val).sum().item()
    num_val_samp = len(labels_val)
    print('End validation - {} samples'.format(num_val_samp))
    return (correct, total)

#%% HYPERPARAMETERS
H = 30
lr = 1e-2 #learning rate
beta = 0.9 #momentum parameter
n_epoch_max = 100 #maximum number of epoch
input_dim = 784

batch_size = 128

#%% Define model, loss and optimizer
model = utils.MLP(H, 10, input_dim)
optimizer = utils.GradientDescentWithMomentum(model, beta=beta, lr=lr)  

#%% Initial validation and plot
num_batch = vec_train.shape[0]/batch_size 
training_loss_v = []
valid_acc_v = []

(correct_best, total) = validation(vec_val, labels_val, model)
print ('Epoch [{}/{}], Valid Acc: {} %'
           .format(0, n_epoch_max, 100 * correct_best / total))
valid_acc_v.append(correct_best / total)

it_v = []
epoch_v = [0]
fig3, axs3 = plt.subplots(ncols=2)
line_loss, = axs3[0].plot(it_v,training_loss_v)
axs3[0].legend()
axs3[0].set_xlabel('Iterations')
line_acc, = axs3[1].plot(epoch_v,valid_acc_v,'b',label='Validation accuracy')
axs3[1].legend()
axs3[1].set_xlabel('Epochs')

it=0
for epoch in range(n_epoch_max):    
    epoch_v.append(epoch+1)
    
    loss_tot = 0
    ids = np.random.permutation(n_train)

    for i in range(n_train//batch_size):
        vec_batch = vec_train[ids[i*batch_size:(i+1)*batch_size],:]
        labels_batch = labels_train[ids[i*batch_size:(i+1)*batch_size]]

        #Forward Pass
        X0,X1,X2,S = model.forward(vec_batch)
        
        #Compute Loss
        l, dc_dS = utils.crossEntropyLoss(S, labels_batch)
        
        #Print Loss
        training_loss_v.append(l)
        it_v.append(it)
        it += 1
        
       
        loss_tot += l.item()
        if (i+1) % 10 == 0:
            line_loss.remove()
            line_loss, = axs3[0].plot(it_v,training_loss_v,'r',label='Training loss')
            axs3[0].legend()
            fig3.canvas.draw()
            fig3.canvas.flush_events()
            #plt.pause(0.1)
            
            #print ('Epoch [{}/{}], Step [{}/{}], Batch Loss: {:.4f}' 
            #       .format(epoch+1, n_epoch_max, i+1, num_batch, l.item()/len(labels)))
    
        #Backward Pass (Compute Gradient)
        optimizer.zero_grad()
        model.backward(dc_dS, S, X2, X1, X0)
        
        #Update Parameters
        optimizer.step()    
        
    
    (correct, total) = validation(vec_val, labels_val, model)
    print ('Epoch [{}/{}], Training Loss: {:.4f}, Valid Acc: {} %'
           .format(epoch+1, n_epoch_max, loss_tot/vec_train.shape[0], 100 * correct / total))
    valid_acc_v.append(correct / total)
    
    
    line_acc.remove()
    line_acc, = axs3[1].plot(epoch_v,valid_acc_v,'b',label='Validation accuracy')
    axs3[1].legend()
    fig3.canvas.draw()
    fig3.canvas.flush_events()
    
    if(correct > correct_best): #early stopping
        correct_best = correct
        #TODO save parameters
        print('Saving model : {}% valid accuracy'.format(100 * correct / total))