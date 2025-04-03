import numpy as np
import matplotlib.pyplot as plt
import utils_autograd as utils
import torch
from MNISTDataset import MNISTDataset
import torchvision.transforms as T


torch.random.manual_seed(0)
path_MNIST_train = '/tmp/MNIST/Training'
mean_norm = 0.1306
std_norm = 0.3081            
training_set = MNISTDataset(path_MNIST_train, mean_norm=mean_norm, std_norm=std_norm)

#%% Show 4 pairs of data
fig1, axs1 = plt.subplots(ncols=4)

offset = 7000
for i in range(4):
    image, label, _ = training_set[i+offset]
    axs1[i].imshow(T.ToPILImage()((image*std_norm)+mean_norm))
    axs1[i].set_title('True label {}'.format(label))
    
plt.pause(1.)

batch_size = 128
train_loader = torch.utils.data.DataLoader(dataset = training_set,
                                       batch_size=batch_size, #nombre d'éléments d'un minibatch
                                       shuffle=True, #mélanger la base de données à la fin de chaque epoch
                                       num_workers=2) #nombre de processus dédiées à la préparation des minibatches

path_MNIST_valid = '/tmp/MNIST/Validation'                
valid_set = MNISTDataset(path_MNIST_valid)
valid_loader = torch.utils.data.DataLoader(dataset = valid_set,
                                       batch_size=2048,
                                       shuffle=False,#inutile de mélanger pour la validation
                                       num_workers=2)

#%% Validation function
def validation(valid_loader, model):
    print('Start validation')
    # Test the model 
    total = 0
    correct = 0
    num_val_samp = 0
    for i, (images, labels, _) in enumerate(valid_loader):
        outputs = model.forward(images.view(-1,784))
        _, predicted = torch.max(outputs[3].data, 1)
        total += len(labels)
        correct += (predicted == labels).sum().item()
        num_val_samp += len(labels)
        
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
training_loss_v = []
valid_acc_v = []

(correct_best, total) = validation(valid_loader, model)
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

    for i, (images, labels, _) in enumerate(train_loader):
        vec_batch = images.view(-1,784)
        labels_batch = labels

        #Forward Pass
        X0,X1,X2,S = model.forward(vec_batch)
        
        #Compute Loss
        l, dc_dS = utils.crossEntropyLoss(S, labels_batch)
        
        #Print Loss
        training_loss_v.append(l.data.numpy())
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
        S.backward(dc_dS)
        
        #Update Parameters
        optimizer.step()    
        
    
    (correct, total) = validation(valid_loader, model)
    print ('Epoch [{}/{}], Training Loss: {:.4f}, Valid Acc: {} %'
           .format(epoch+1, n_epoch_max, loss_tot/len(train_loader), 100 * correct / total))
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