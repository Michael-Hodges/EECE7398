#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torchvision
import argparse

cuda = torch.device('cuda:0') 
# ----------------- prepare training data -----------------------
transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Visualize feature maps
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        #input size HxWxK (32x32x3)
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        										#output size
        self.conv11 = nn.Conv2d(3, 32, 5) 		#(28x28x32)
        self.conv12 = nn.Conv2d(32,32,1)
        self.conv13 = nn.Conv2d(32,32,1)
        self.mp1 = nn.MaxPool2d(3, stride = 1) 	#(26x26x32)
        # self.mp11 = nn.MaxPool2d(4, stride =2)
        self.bn2d1 = nn.BatchNorm2d(32)
        self.conv21 = nn.Conv2d(32,64,3) 		#(24x24x64)
        self.conv22 = nn.Conv2d(64,64,1)
        self.conv23 = nn.Conv2d(64,64,1)
        self.mp2 = nn.MaxPool2d(4, stride = 2)  #(10x10x16)
        self.bn2d2 = nn.BatchNorm2d(64)
        self.conv31 = nn.Conv2d(64,128,3)		#(8x8x128)
        self.conv32 = nn.Conv2d(128,128,1)
        self.conv33 = nn.Conv2d(128,128,1)
        self.bn2d3 = nn.BatchNorm2d(128)
        self.mp3 = nn.MaxPool2d(3, stride = 2)	#(5x5z128)

        # self.conv31 = nn.Conv2d(16,16,3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(4 * 4 * 128, 2048)  # 5x5x16 from image mp2
        self.fc1_bn = nn.BatchNorm1d(2048)
        self.fc2 = nn.Linear(2048, 10)
        self.fc2_bn = nn.BatchNorm1d(10)
        self.fc3 = nn.Linear(64,10)

    def forward(self, x):
        x = F.relu(self.conv11(x))
        x = self.bn2d1(x)
        x = self.mp1(x)
        x = F.relu(self.conv21(x))
        x = self.bn2d2(x)
        x = self.mp2(x)
        x = F.relu(self.conv31(x))
        x = self.bn2d3(x)
        x = self.mp3(x)
        # x = self.conv31(x)
        x = x.view(-1, 4*4*128)
        x = self.fc1_bn(F.relu(self.fc1(x)))
        x = F.softmax(self.fc2(x), dim = -1)
        return x


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('input', nargs='?')
	parser.add_argument('files', nargs=argparse.REMAINDER)
	args = parser.parse_args()
	net = Net()
	net.cuda()
	if args.input == 'train':
		train_data = torchvision.datasets.CIFAR10(
		    root='./data.cifar10',                          # location of the dataset
		    train=True,                                     # this is training data
		    transform=transform,    # Converts a PIL.Image or numpy.ndarray to torch.FloatTensor of shape (C x H x W)
		    download=True                                   # if you haven't had the dataset, this will automatically download it for you
		)

		train_loader = Data.DataLoader(dataset=train_data, batch_size=256, shuffle=True, 
		    sampler=None, batch_sampler=None, num_workers=2, collate_fn=None,
		    pin_memory=True, drop_last=False, timeout=0,
		    worker_init_fn=None)
	#	train_loader = train_loader.to('cuda')

		test_data = torchvision.datasets.CIFAR10(
		    root='./data.cifar10/', 
		    train=False, 
		    transform=transform,
		    download=True
		)

		test_loader = Data.DataLoader(dataset=test_data, batch_size=256,
		    shuffle=False, num_workers=2, pin_memory=True)

	#	test_loader = test_loader.to('cuda')

	if args.input == 'train':
		loss_func = nn.CrossEntropyLoss()
		optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=0.0)

		print('{0}, {1}, {2}, {3}, {4}'.format('Epoch', 'Train Loss', 'Train Acc %', 'Test Loss', 'Test Acc %'))
		epoch = 0
		prev_test_accuracy = 0
		test_accuracy = 0
		epoch_delta = -999
		#for epoch in range(100):
		while (np.abs(epoch_delta)>.1) and (epoch < 100):
		    running_loss = 0.0
		    train_accuracy = 0.0
		    t_correct = 0
		    total = 0
		    for step, (inputs, target) in enumerate(train_loader):
		        net.train()   # set the model in training mode
		        inputs, target = inputs.to(cuda), target.to(cuda)
		        # print(inputs.device)
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
			        images, labels = images.to(cuda), labels.to(cuda)
			        # print(images.size())
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

		class_correct = list(0. for i in range(10))
		class_total = list(0. for i in range(10))
		with torch.no_grad():
			for data in test_loader:
				images, labels = data
				images, labels = images.to(cuda), labels.to(cuda)
				outputs = net(images)
				_, predicted = torch.max(outputs, 1)
				c = (predicted == labels).squeeze()
				for i in range(4):
					label = labels[i]
					class_correct[label] += c[i].item()
					class_total[label] += 1
		for i in range(10):
			print('Accuracy of %5s : %2d %%' % (
				classes[i], 100 * class_correct[i] / class_total[i]))

		print("Training terminated. Saving model...")
		torch.save(net.state_dict(), "./model/cifar_CNN_2.pt")
	
	if args.input == 'test':
		net.load_state_dict(torch.load("./model/cifar_CNN_2.pt"))
		net.eval()
		net.conv11.register_forward_hook(get_activation('conv11'))
		image = Image.open(args.files[0])
		tensImage = transform(image)
		# tensImage = Variable(tensImage, requires_grad=True)
		tensImage = tensImage.unsqueeze(0)
		output = net(tensImage)
		
		act = activation['conv11'].squeeze()
		fig = plt.figure(figsize = (np.ceil(np.sqrt(act.size(0))), np.ceil(np.sqrt(act.size(0)))))
		grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(int(np.ceil(np.sqrt(act.size(0)))), int(np.ceil(np.sqrt(act.size(0))))),  # creates 2x2 grid of axes
                 axes_pad=0.0, label_mode = "1"  # pad between axes in inch.
                 )
		for ax, im in zip(grid, act):
			ax.imshow(im)
		plt.savefig("CONV_rslt.png", dpi=150)
		_, prediction = torch.max(output.data,1)
		print("prediction result: {}".format(classes[prediction]))
		
		# test_data = torchvision.datasets.CIFAR10(
		#     root='./data.cifar10/', 
		#     train=False, 
		#     transform=transform,
		#     download=True
		# )

		# test_loader = Data.DataLoader(dataset=test_data, batch_size=256,
		#     shuffle=False, num_workers=2)
		
		# class_correct = list(0. for i in range(10))
		# class_total = list(0. for i in range(10))
		# with torch.no_grad():
		# 	for data in test_loader:
		# 		images, labels = data
		# 		outputs = net(images)
		# 		_, predicted = torch.max(outputs, 1)
		# 		c = (predicted == labels).squeeze()
		# 		for i in range(4):
		# 			label = labels[i]
		# 			class_correct[label] += c[i].item()
		# 			class_total[label] += 1
		# for i in range(10):
		# 	print('Accuracy of %5s : %2d %%' % (
		# 		classes[i], 100 * class_correct[i] / class_total[i]))



if __name__ == '__main__':
	main()
