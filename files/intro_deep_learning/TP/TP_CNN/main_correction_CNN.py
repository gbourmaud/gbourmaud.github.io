
import torch
from MNISTDataset import MNISTDataset
import matplotlib.pyplot as plt
import torch.nn as nn
import math
from torch.utils.tensorboard import SummaryWriter


class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU())
        self.fc = nn.Linear(7*7*32, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.fc(out.flatten(start_dim = 1))
        return out
    
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

torch.random.manual_seed(0)

path_MNIST_train = '/tmp/MNIST/Training'
mean_norm = 0.1306
std_norm = 0.3081            
training_set = MNISTDataset(path_MNIST_train, mean_norm = mean_norm, std_norm = std_norm)


#%% Train loader
batch_size = 128
train_loader = torch.utils.data.DataLoader(dataset = training_set,
                                       batch_size=batch_size, #nombre d'éléments d'un minibatch
                                       shuffle=True, #mélanger la base de données à la fin de chaque epoch
                                       num_workers=2) #nombre de processus dédiées à la préparation des minibatches



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
            labels = labels.to(device)
            
            outputs = model(images.to(device))
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
model = CNN(10).to(device) #moves the model to GPU if available
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=beta)  
criterion = nn.CrossEntropyLoss()

writer = SummaryWriter()

#%% Initial validation and plot
num_batch = len(train_loader) 

(correct_best, total) = validation(valid_loader, model, device)
print ('Epoch [{}/{}], Valid Acc: {} %'
           .format(0, n_epoch_max, 100 * correct_best / total))

writer.add_scalar("Accuracy/valid", 100 * correct_best / total, 0)

it=0
for epoch in range(n_epoch_max):    
    
    loss_tot = 0

    for i, (images, labels,_) in enumerate(train_loader):

        it += 1
        # Reshape images to (batch_size, input_size), actual shape is (batch_size, 1, 28, 28)
        
        #Forward Pass
        S = model.forward(images.to(device))
        
        #Compute Loss
        labels.to(device)
        l = criterion(S, labels)
                
        writer.add_scalar("Training_loss", l.data, it)
        
        loss_tot += l.item()
    
        #Backward Pass (Compute Gradient)
        optimizer.zero_grad()
        l.backward()
        
        #Update Parameters
        optimizer.step()    
        
    
    (correct, total) = validation(valid_loader, model, device)
    print ('Epoch [{}/{}], Training Loss: {:.4f}, Valid Acc: {} %'
           .format(epoch+1, n_epoch_max, loss_tot/len(training_set), 100 * correct / total))
    
    writer.add_scalar("Accuracy/valid", 100 * correct / total, epoch+1)
    writer.flush()

    
    if(correct > correct_best): #early stopping
        correct_best = correct
        torch.save(model.state_dict(), './model.pt')
        print('Saving model : {}% valid accuracy'.format(100 * correct / total))
    
writer.close()