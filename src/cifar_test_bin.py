import pandas as pd 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import pickle
import torch.optim as optim


def run_full_code_bin(device, model_file, test_data_file, output_file, n):

	def unpickle(file):
		with open(file, 'rb') as fo:
			dict = pickle.load(fo, encoding='bytes')
		return dict

	def normalise(X):
		return (X - X.mean(axis = 0))/(X.std(axis = 0) + (np.ones((1,X.shape[1]))*(1e-06)))
	
	dict6 = unpickle(test_data_file)
	Xtest = np.array(dict6[b'data'])
	# Ytest = np.array(dict6[b'labels'])
	Xtest = normalise(Xtest)
	Xtest = Xtest.reshape(10000, 3, 32, 32)
	Xtest  = torch.from_numpy(Xtest)
	# Ytest = Ytest.astype(int)
	# Ytest = torch.from_numpy(Ytest)
	Xtest = Xtest.to(torch.float32)
	# Ytest = Ytest.to(torch.int64)

	class LambdaLayer(nn.Module):
		def __init__(self, lambd):
			super(LambdaLayer, self).__init__()
			self.lambd = lambd

		def forward(self, x):
			return self.lambd(x)

	class Batch_Normalisation(nn.Module):
		def __init__(self, numlayer, momentum1 = 0.1):
			super().__init__()
			self.gamma = nn.Parameter(torch.ones((1, numlayer, 1, 1)), requires_grad  = True)
			self.beta = nn.Parameter(torch.zeros((1, numlayer, 1, 1)), requires_grad  = True)
			self.mmean = nn.Parameter(torch.zeros((1, numlayer, 1, 1)).to(torch.float32), requires_grad = False)
			self.mvar = nn.Parameter(torch.ones((1, numlayer, 1, 1)).to(torch.float32), requires_grad = False)
			self.momentum = momentum1
			self.eps = 1e-6
		def forward(self, x):
			self.mmean = self.mmean.to(x.device)
			self.mvar = self.mvar.to(x.device)
			if torch.is_grad_enabled():
				mean = x.mean(dim = (0, 2, 3), keepdim=True)
				var = ((x - mean)**2).mean(dim = (0, 2, 3), keepdim=True)
				var = ((x.shape[0])/(x.shape[0] - 1))*var
				x = (x - mean) / torch.sqrt(var + self.eps)
				self.mmean = nn.Parameter(self.momentum * self.mmean + (1.0 - self.momentum) * mean, requires_grad = False)
				self.mvar = nn.Parameter(self.momentum * self.mvar + (1.0 - self.momentum) * var, requires_grad = False)
			else:  
				x = (x - self.mmean) / torch.sqrt(self.mvar + self.eps)        
			x = self.gamma * x + self.beta  
			return x
	
	class Instance_Normalisation(nn.Module):
		def __init__(self, numlayer):
			super().__init__()
			self.gamma = nn.Parameter(torch.ones((1, numlayer, 1, 1)), requires_grad  = True)
			self.beta = nn.Parameter(torch.zeros((1, numlayer, 1, 1)), requires_grad  = True)
			self.eps = 1e-6
		def forward(self, x):
			mean = x.mean(dim = (2, 3), keepdim=True)
			var = ((x - mean)**2).mean(dim = (2, 3), keepdim=True)
			x = (x - mean) / torch.sqrt(var + self.eps)
			x = self.gamma * x + self.beta  
			return x
	
	class Batch_Instance_Normalisation(nn.Module):
		def __init__(self, numlayer):
			super(Batch_Instance_Normalisation, self).__init__()
			self.rho = nn.Parameter(0.5*torch.ones((1, numlayer, 1, 1)), requires_grad  = True)
			self.batch_norm = Batch_Normalisation(numlayer, 0.1)
			self.instance_norm = Instance_Normalisation(numlayer)
		def forward(self, x):
			self.rho = nn.Parameter(torch.clamp(self.rho, min = 0, max = 1))
			y = x
			x = self.batch_norm(x)
			y = self.instance_norm(y)
			x = self.rho*x + (1-self.rho)*y
			return x
	
	class ResNetBlock(nn.Module):
		def __init__(self, numlayer, n):
			super(ResNetBlock, self).__init__()
			self.conv1 = nn.Conv2d(numlayer, numlayer, 3, padding = 1)
			self.batch_instance_norm1 = Batch_Instance_Normalisation(numlayer)
			self.conv2 = nn.Conv2d(numlayer, numlayer, 3, padding = 1)
			self.batch_instance_norm2 = Batch_Instance_Normalisation(numlayer)
		def forward(self, x):
			y = x
			x = self.conv1(x)
			x = self.batch_instance_norm1(x)
			x = F.relu(x)
			x = self.conv2(x)
			x = self.batch_instance_norm2(x)
			x = x + y
			x = F.relu(x);
			return x

	class ResNet_Layer(nn.Module):
		def __init__(self, numlayer, n):
			super(ResNet_Layer, self).__init__()
			self.conv_blocs = nn.Sequential(*[ResNetBlock(numlayer, n)
				for i in range(0, n)])
		def forward(self, x):
			x = self.conv_blocs(x);
			return x

	class ResNet_Downsample(nn.Module):
		def __init__(self, numlayerin, numlayerout, n):
			super(ResNet_Downsample, self).__init__()
			self.conv1 = nn.Conv2d(numlayerin, numlayerout, 3, stride = 2, padding = 1)
			self.batch_instance_norm1 = Batch_Instance_Normalisation(numlayerout)
			self.conv2 = nn.Conv2d(numlayerout, numlayerout, 3, padding = 1)
			self.batch_instance_norm2 = Batch_Instance_Normalisation(numlayerout)
			self.s1A = LambdaLayer(lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, int(numlayerin/2), int(numlayerin/2)), "constant", 0))
		def forward(self, x):
			y = x
			x = self.conv1(x)
			x = self.batch_instance_norm1(x)
			x = F.relu(x)
			x = self.conv2(x)
			x = self.batch_instance_norm2(x)
			x = x + self.s1A(y)
			x = F.relu(x)
			return x

	class ResNet(nn.Module):
		def __init__(self, n1, r1):
			super(ResNet, self).__init__()
			self.n = n1
			self.r = r1
			self.conv_3_16 = nn.Conv2d(3, 16, 3, padding = 1)
			self.batch_instance_norm1 = Batch_Instance_Normalisation(16)
			self.resnet_layer1 = ResNet_Layer(16, n1)
			self.resnet_block1 = ResNet_Downsample(16, 32, n1)
			self.resnet_layer2 = ResNet_Layer(32, n1-1)
			self.resnet_block2 = ResNet_Downsample(32, 64, n1)
			self.resnet_layer3 = ResNet_Layer(64, n1-1)
			self.globalAvg = nn.AdaptiveAvgPool2d((1, 1))
			self.fc1 = nn.Linear(64, self.r)
		def forward(self, x):
			x =  self.conv_3_16(x)
			x = self.batch_instance_norm1(x)
			x = F.relu(x)
			x = self.resnet_layer1(x)
			x = self.resnet_block1(x)
			x = self.resnet_layer2(x)
			x = self.resnet_block2(x)
			x = self.resnet_layer3(x)
			#Global average pooling
			x = self.globalAvg(x)

			y = x.view(-1, 64)
			x = self.fc1(y)   
			return x, y  

	model = ResNet(n, 10)
	model.load_state_dict(torch.load(model_file))
	model = model.to(device)

	len_Xtest = Xtest.shape[0]
	final_preds = np.array([]).reshape((0, 10))
	batch_size = 128
	for i in range(0, (len_Xtest//batch_size)):
		x = torch.FloatTensor(Xtest[i*batch_size:(i+1)*batch_size, :]).to(device)
		with torch.no_grad():    
			preds, _ = model(x)
			final_preds = np.vstack((final_preds, preds.detach().cpu().numpy()))  
	if(len_Xtest - ((len_Xtest//batch_size)*batch_size) > 0):
		x = torch.FloatTensor(Xtest[((len_Xtest//batch_size)*batch_size):len_Xtest, :]).to(device)
		with torch.no_grad():    
			preds, _ = model(x)
			final_preds = np.vstack((final_preds, preds.detach().cpu().numpy()))
	print(final_preds.shape)
	final_preds = final_preds.argmax(axis = 1)
	final_preds = final_preds.reshape(final_preds.shape[0])


	# # get predictions for val data
	# with torch.no_grad():
	# 	preds, _ = model(Xtest.to(device))
	# 	preds = preds.detach().cpu().numpy()
	# # prediction
	# prediction = preds.argmax(axis = 1)
	s = ""
	for x in final_preds:
		s += str(x) + "\n"
	with open(output_file, "w") as f:
		f.write(s)

