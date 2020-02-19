#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torchvision
import argparse

# ----------------- prepare training data -----------------------
transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(3 * 32 * 32, 5120)  # 3*32*32 from image dimension
        self.fc1_bn = nn.BatchNorm1d(5120)
        self.fc2 = nn.Linear(5120, 5120)
        self.fc2_bn = nn.BatchNorm1d(5120)
        self.fc3 = nn.Linear(5120, 3072)
        self.fc3_bn = nn.BatchNorm1d(3072)
        self.fc4 = nn.Linear(3072, 2048)
        self.fc4_bn = nn.BatchNorm1d(2048)
        self.fc5 = nn.Linear(2048, 1024)
        self.fc5_bn = nn.BatchNorm1d(1024)
        self.fc6 = nn.Linear(1024, 512)
        self.fc6_bn = nn.BatchNorm1d(512)
        self.fc7 = nn.Linear(512, 256)
        self.fc7_bn = nn.BatchNorm1d(256)
        self.fc8 = nn.Linear(256, 10)
        # self.conv1 = nn.Conv2d(3, 6, 5)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(6, 16, 5)
        # self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = x.view(-1, 3*32*32)
        x = self.fc1_bn(F.relu(self.fc1(x)))
        x = self.fc2_bn(F.relu(self.fc2(x)))
        x = self.fc3_bn(F.relu(self.fc3(x)))
        x = self.fc4_bn(F.relu(self.fc4(x)))
        x = self.fc5_bn(F.relu(self.fc5(x)))
        x = self.fc6_bn(F.relu(self.fc6(x)))
        x = self.fc7_bn(F.relu(self.fc7(x)))
        x = F.softmax(self.fc8(x), dim = -1)
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        # x = x.view(-1, 16 * 5 * 5)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('input', nargs='?')
	parser.add_argument('files', nargs=argparse.REMAINDER)
	args = parser.parse_args()
	net = Net()

	if args.input == 'train':
		train_data = torchvision.datasets.CIFAR10(
		    root='./data.cifar10',                          # location of the dataset
		    train=True,                                     # this is training data
		    transform=transform,    # Converts a PIL.Image or numpy.ndarray to torch.FloatTensor of shape (C x H x W)
		    download=True                                   # if you haven't had the dataset, this will automatically download it for you
		)

		train_loader = Data.DataLoader(dataset=train_data, batch_size=256, shuffle=True, 
		    sampler=None, batch_sampler=None, num_workers=2, collate_fn=None,
		    pin_memory=False, drop_last=False, timeout=0,
		    worker_init_fn=None)

		test_data = torchvision.datasets.CIFAR10(
		    root='./data.cifar10/', 
		    train=False, 
		    transform=transform,
		    download=True
		)

		test_loader = Data.DataLoader(dataset=test_data, batch_size=256,
		    shuffle=False, num_workers=2)


	if args.input == 'train':
		loss_func = nn.CrossEntropyLoss()
		optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=0.0)

		print('{0}, {1}, {2}, {3}, {4}'.format('Epoch', 'Train Loss', 'Train Acc %', 'Test Loss', 'Test Acc %'))
		epoch = 0
		prev_test_accuracy = 0
		epoch_delta = -999
		#for epoch in range(100):
		while np.abs(epoch_delta)>.05:
		    running_loss = 0.0
		    train_accuracy = 0.0
		    t_correct = 0
		    total = 0
		    for step, (inputs, target) in enumerate(train_loader):
		        net.train()   # set the model in training mode
		        # zero the parameter gradients
		        optimizer.zero_grad()
		        
		        # forward + backward + optimize
		        outputs = net(inputs)
		        loss = loss_func(outputs, target)
		        loss.backward()
		        optimizer.step()
		        running_loss += loss.item()/target.size(0)
		        _, predicted = torch.max(outputs.data, 1)
		        total += target.size(0)
		        t_correct += (predicted == target).sum().item()
		        train_accuracy = 100 * t_correct / total

		    test_accuracy = 0
		    test_correct = 0
		    test_total = 0
		    test_running_loss = 0
		    with torch.no_grad():
		        for data in test_loader:
			        images, labels = data
			        outputs = net(images)
			        test_loss = loss_func(outputs, labels)
			        test_running_loss += test_loss.item()/labels.size(0)
			        _, predicted = torch.max(outputs.data, 1)
			        test_total += labels.size(0)
			        test_correct += (predicted == labels).sum().item()
		    test_accuracy = 100 * test_correct / test_total
		    epoch_delta = test_accuracy - prev_test_accuracy
		    prev_test_accuracy = test_accuracy
		    epoch += 1
		    print('{0:5d}, {1:10.3f}, {2:11.3f}, {3:9.3f}, {4:10.3f}'.format(epoch, running_loss, train_accuracy, test_running_loss, test_accuracy))

		print("Training terminated. Saving model...")
		torch.save(net.state_dict(), "./model/cifar_nn_1.pt")
	
	if args.input == 'test':
		net.load_state_dict(torch.load("./model/cifar_nn.pt"))
		net.eval()
		image = Image.open(args.files[0])
		tensImage = transform(image)

		output = net(tensImage)
		_, prediction = torch.max(output.data,1)
		print("prediction result: {}".format(classes[prediction]))
		

if __name__ == '__main__':
	main()
















# def __init(self, n_inp, n_hidden, n_outp):
# 	superl(Net,self).__init__()
# 	self.hidden = torch.nn.Linear(n_inp, n_hidden)
# 	self.outp = torch.nn.Linear(n_hidden, n_outp)

# def forard(self, x):
# 	x = F.relu(self.hidden(x))
# 	x = self.outp(x)
# 	return x


# model = Net(1, 10, 1) # will be useful for scanner model parameters
