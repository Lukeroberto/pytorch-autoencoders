import os
import matplotlib.pyplot as plt
import numpy as np

import torch 
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms

root = './data'
if not os.path.exists(root):
    os.mkdir(root)

img = './results/convolutional'
if not os.path.exists(img):
    os.mkdir(img)

learning_rate = 1e-3
batch_size = 128
num_epochs = range(50)

trans = transforms.Compose([transforms.ToTensor()])
train_set = dset.MNIST(root=root, train=True, transform=trans, download=True)

train_loader = torch.utils.data.DataLoader(
                dataset=train_set,
                batch_size=batch_size,
                shuffle=True)

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def name(self):
        return "ConvAutoEncoder"

model = AutoEncoder()
use_cuda = torch.cuda.is_available()
if use_cuda:
    model = model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

def mse_loss(input, target):
    return torch.sum((input - target) ** 2)

print("Training")
for epoch in num_epochs:

    # Training
    print("Epoch:", epoch)
    for batch_idx, (x, target) in enumerate(train_loader):

        optimizer.zero_grad()

        if use_cuda:
            x = x.cuda()

        x = Variable(x)
        output = model(x)

        loss = mse_loss(output, x)

        loss.backward()
        optimizer.step()

        if (batch_idx+1) % 100 == 0 or (batch_idx+1) == len(train_loader):
            print("Batch Index: {}, Train Loss: {:.6f}".format(batch_idx+1, loss))
    
    print("Saving last example")
    fig, (ax1, ax2) = plt.subplots(1, 2)

    input = x.detach()
    out = model(x).detach()

    if use_cuda:
        input = input.cpu()
        out = out.cpu()

    ax1.imshow(np.squeeze(input[0, :, :, :]))
    ax2.imshow(np.squeeze(out[0, :, :, :]))

    plt.savefig(img + "/train_target_{}_epoch_{}.png".format(target[0], epoch))

# Save Model
torch.save(model.state_dict(), img + "/" + model.name())

    
