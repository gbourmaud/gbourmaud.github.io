from MLP import MLP
import torch
import matplotlib.pyplot as plt
from MNISTDataset import MNISTDataset
import torchvision.transforms as T


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

H = 30
model = MLP(H).to(device) #moves the model to GPU if available

model.load_state_dict(torch.load('model.pt', weights_only=True))
model.eval()


path_MNIST_test = '/tmp/MNIST/Testing'                
test_set = MNISTDataset(path_MNIST_test)
test_loader = torch.utils.data.DataLoader(dataset = test_set,
                                       batch_size=128,
                                       shuffle=False,#inutile de mélanger pour le test
                                       num_workers=2)

model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels, _ in test_loader:
        images_vec = images.view(-1, 28*28).to(device)
        labels = labels.to(device)
        
        outputs = model(images_vec)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print ('Test Acc: {} %'
        .format(100 * correct / total))

#%% plot results

img,_,_ = test_set[3598]
model.eval()
outputs = model(img.unsqueeze(0).view(-1, 28*28).to(device))
_, predicted = torch.max(outputs.data, 1)
fig,ax = plt.subplots()
ax.imshow(T.ToPILImage()(img))
ax.set_title('Predicted class : {}'.format(predicted[0]))