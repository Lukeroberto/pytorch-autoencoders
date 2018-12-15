import os
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

root = './data'
if not os.path.exists(root):
    os.mkdir(root)

img = './results/fully_connected'
if not os.path.exists(img):
    os.mkdir(img)

trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])

train_set = dset.MNIST(root=root, train=True, transform=trans, download=True)
test_set  = dset.MNIST(root=root, train=False, transform=trans, download=True)

batch_size = 128

train_loader = torch.utils.data.DataLoader(
                dataset=train_set, 
                batch_size=batch_size,
                shuffle=True)

test_loader = torch.utils.data.DataLoader(
                dataset=test_set, 
                batch_size=batch_size,
                shuffle=True)

# Network
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        # 3 passes down to 100 dims
        # 3 passes back up to 28*28
        self.fc1 = nn.Linear(28*28, 500)
        self.fc2 = nn.Linear(500, 256)
        self.fc3 = nn.Linear(256, 100)

        self.fc4 = nn.Linear(100, 256)
        self.fc5 = nn.Linear(256, 500)
        self.fc6 = nn.Linear(500, 28*28)
    
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        return x

    def name(self):
        return "AutoEncoder"

model = AutoEncoder()

use_cuda = torch.cuda.is_available()
if use_cuda:
    model = model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

def mse_loss(input, target):
    return torch.sum((input - target) ** 2)

for epoch in range(10):

    # Training
    print("Training")
    for batch_idx, (x, target) in enumerate(train_loader):

        optimizer.zero_grad()

        if use_cuda:
            x = x.cuda()

        x = Variable(x)
        output = model(x)

        loss  = mse_loss(output, x.view(-1, 28*28))

        loss.backward()
        optimizer.step()

        if (batch_idx+1) % 100 == 0 or (batch_idx+1) == len(train_loader):
            print("Epoch: {}, Batch index: {}, Train Loss: {:.6f}".format(epoch, batch_idx+1, loss))
    
    
    print("Train Results")
    fig, (ax1, ax2) = plt.subplots(1, 2)
    
    input = x.view(-1, 28, 28, 1).detach()
    out = model(x).view(-1, 28, 28, 1).detach()
    
    if use_cuda:
        input = input.cpu()
        out = out.cpu()

    ax1.imshow(np.squeeze(input[0, :, :, :]))
    ax2.imshow(np.squeeze(out[0, :, :, :]))
    
    plt.savefig(img + "/train_target_{}_epoch_{}.png".format(target[0],epoch))

# Save Model
torch.save(model.state_dict(), img + "/" + model.name())
