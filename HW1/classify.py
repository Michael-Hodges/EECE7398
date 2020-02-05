#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torchvision
import argparse

# ----------------- prepare training data -----------------------
train_data = torchvision.datasets.CIFAR10(
    root='./data.cifar10',                          # location of the dataset
    train=True,                                     # this is training data
    transform=torchvision.transforms.ToTensor(),    # Converts a PIL.Image or numpy.ndarray to torch.FloatTensor of shape (C x H x W)
    download=True                                   # if you haven't had the dataset, this will automatically download it for you
)

train_loader = Data.DataLoader(dataset=train_data, batch_size=128, shuffle=True, 
    sampler=None, batch_sampler=None, num_workers=2, collate_fn=None,
    pin_memory=False, drop_last=False, timeout=0,
    worker_init_fn=None)

test_data = torchvision.datasets.CIFAR10(
    root='./data.cifar10/', 
    train=False, 
    transform=torchvision.transforms.ToTensor(),
    download=True
)

test_loader = Data.DataLoader(dataset=test_data, batch_size=128,
    shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(train_loader)
images, labels = dataiter.next()

# show images
#imshow(torchvision.utils.make_grid(images))
# print labels
#print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        #self.conv1 = nn.Conv2d(1, 6, 3)
        #self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(3 * 32 * 32, 120)  # 32*32 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = x.view(-1, self.num_flat_features(x))
        # print(x.size()) # [batch size, pixels 3x32x32] 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim = -1)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

parser = argparse.ArgumentParser()
parser.add_argument('input', nargs='?')
args = parser.parse_args()
if args.input == 'train':
	print('train')
elif args.input == 'test':
	print('test')
net = Net()
print(net)


loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters())

print('{0}, {1}, {2}, {3}, {4}'.format('Epoch', 'Train Loss', 'Train Acc %', 'Test Loss', 'Test Acc %'))
for epoch in range(100):
    running_loss = 0.0
    for step, (inputs, target) in enumerate(train_loader):
        net.train()   # set the model in training mode
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward + backward + optimize
        outputs = net(inputs)
        loss = loss_func(outputs, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    # print statistics
    print('{0:5d}, {1:10.3f}, {2:11.3f}, {3:9.3f}, {4:10.3f}'.format(epoch+1,running_loss, 0, 0, 0))
#         # if step % 50 == 0:
#         #     test()
#         #     ...
#         #     ...
#         #     save_model()
#         #     ...
#         #     ...




















# def __init(self, n_inp, n_hidden, n_outp):
# 	superl(Net,self).__init__()
# 	self.hidden = torch.nn.Linear(n_inp, n_hidden)
# 	self.outp = torch.nn.Linear(n_hidden, n_outp)

# def forard(self, x):
# 	x = F.relu(self.hidden(x))
# 	x = self.outp(x)
# 	return x


# model = Net(1, 10, 1) # will be useful for scanner model parameters
