#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 10:16:28 2022

@author: guillaume
"""
import torch
from MNISTDataset import MNISTDataset
import torchvision.transforms as T
import matplotlib.pyplot as plt
import torch.nn as nn
from MLP import MLP

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

torch.random.manual_seed(0)

path_MNIST_train = '/tmp/MNIST/Training'
mean_norm = 0.1306
std_norm = 0.3081            
training_set = MNISTDataset(path_MNIST_train, mean_norm = mean_norm, std_norm = std_norm)

# #%% Show 4 pairs of data
# fig1, axs1 = plt.subplots(ncols=4)

# offset = 7000
# for i in range(4):
#     image, label, _ = training_set[i+offset]
#     axs1[i].imshow(T.ToPILImage()((image*std_norm)+mean_norm))
#     axs1[i].set_title('True label {}'.format(label))
    
# plt.pause(1.)

# #%% Compute mean and std
# mean = 0
# data_full = []
# for i in range(len(training_set)):
#     print(i)
#     image, label, _ = training_set[i]
#     data_full.append(image.ravel().data) #stores all data, only possible because MNIST is small
    
# data_full_concat = torch.concatenate(data_full)
# mean = torch.mean(data_full_concat)
# std = torch.std(data_full_concat)
# print('Mean MNIST {}, Std MNIST {}'.format(mean, std))

#%% Train loader
batch_size = 128
train_loader = torch.utils.data.DataLoader(dataset = training_set,
                                       batch_size=batch_size, #nombre d'éléments d'un minibatch
                                       shuffle=True, #mélanger la base de données à la fin de chaque epoch
                                       num_workers=2) #nombre de processus dédiées à la préparation des minibatches

# images, labels, _ = next(iter(train_loader))

# # Show 10 pairs of data
# fig2, axs2 = plt.subplots(ncols=10)
# for i in range(10):
#     axs2[i].imshow(T.ToPILImage()((images[i,:,:,:]*std_norm)+mean_norm))
#     axs2[i].set_title('{}'.format(labels[i]))
    
# plt.pause(1.)

#%% Valid loader

path_MNIST_valid = '/tmp/MNIST/Validation'                
valid_set = MNISTDataset(path_MNIST_valid)
valid_loader = torch.utils.data.DataLoader(dataset = valid_set,
                                       batch_size=batch_size,
                                       shuffle=False,#inutile de mélanger pour la validation
                                       num_workers=2)

   
        

    
#%% Validation function
def validation(valid_loader, model, device):
    print('Start validation')
    model.eval()
    # Test the model
    # In test phase, we don't need to compute gradients (for memory efficiency)
    num_val_samp = 0
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels, _ in valid_loader:
            images_vec = images.view(-1, 28*28).to(device)
            labels = labels.to(device)
            
            outputs = model(images_vec)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            num_val_samp += len(labels)
    print('End validation - {} samples'.format(num_val_samp))
    model.train()
    return (correct, total)

#%% HYPERPARAMETERS
H = 30
lr = 1e-2 #learning rate
beta = 0.9 #momentum parameter
n_epoch_max = 100 #maximum number of epoch
input_dim = 784

#%% Define model, loss and optimizer
model = MLP(H).to(device) #moves the model to GPU if available
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=beta)  
criterion = nn.CrossEntropyLoss()

#%% Initial validation and plot
num_batch = len(train_loader) 
training_loss_v = []
valid_acc_v = []

(correct_best, total) = validation(valid_loader, model, device)
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

    for i, (images, labels,_) in enumerate(train_loader):

        # Reshape images to (batch_size, input_size), actual shape is (batch_size, 1, 28, 28)
        images_vec = images.view(-1, input_dim).to(device)
        
        #Forward Pass
        S = model.forward(images_vec)
        
        #Compute Loss
        labels.to(device)
        l = criterion(S, labels)
        
        #Print Loss
        training_loss_v.append(l.data)
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
        l.backward()
        
        #Update Parameters
        optimizer.step()    
        
    
    (correct, total) = validation(valid_loader, model, device)
    print ('Epoch [{}/{}], Training Loss: {:.4f}, Valid Acc: {} %'
           .format(epoch+1, n_epoch_max, loss_tot/len(training_set), 100 * correct / total))
    valid_acc_v.append(correct / total)
    
    
    line_acc.remove()
    line_acc, = axs3[1].plot(epoch_v,valid_acc_v,'b',label='Validation accuracy')
    axs3[1].legend()
    fig3.canvas.draw()
    fig3.canvas.flush_events()
    
    if(correct > correct_best): #early stopping
        correct_best = correct
        torch.save(model.state_dict(), './model.pt')
        print('Saving model : {}% valid accuracy'.format(100 * correct / total))
    
    
