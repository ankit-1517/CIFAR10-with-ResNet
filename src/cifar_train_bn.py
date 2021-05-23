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


def run_full_code_bn(device, data_dir, output_file, n):
	def unpickle(file):
		with open(file, 'rb') as fo:
			dict = pickle.load(fo, encoding='bytes')
		return dict

	path = data_dir
	if path.endswith("/") == False:
		path += "/"
	dict1 = unpickle(path + 'data_batch_1')
	dict2 = unpickle(path + 'data_batch_2')
	dict3 = unpickle(path + 'data_batch_3')
	dict4 = unpickle(path + 'data_batch_4')
	dict5 = unpickle(path + 'data_batch_5')

	X = np.vstack((np.array(dict1[b'data']), np.array(dict2[b'data']), np.array(dict3[b'data']), np.array(dict4[b'data']), np.array(dict5[b'data'])))
	Y = np.vstack((np.array(dict1[b'labels']).reshape(-1,1), np.array(dict2[b'labels']).reshape(-1,1), np.array(dict3[b'labels']).reshape(-1,1), np.array(dict4[b'labels']).reshape(-1,1), np.array(dict5[b'labels']).reshape(-1,1)))
	Xtrain, Xval, Ytrain, Yval = train_test_split(X, Y, test_size = 0.2, random_state = 0)

	def normalise(X):
		return (X - X.mean(axis = 0))/(X.std(axis = 0) + (np.ones((1,X.shape[1]))*(1e-06)))
	
	Xtrain = normalise(Xtrain)
	Xval = normalise(Xval)

	Xtrain = Xtrain.reshape(40000, 3, 32, 32)
	Xtrain  = torch.from_numpy(Xtrain)

	Ytrain = Ytrain.astype(int)
	Ytrain = torch.from_numpy(Ytrain)

	Xval = Xval.reshape(10000, 3, 32, 32)
	Xval  = torch.from_numpy(Xval)

	Yval = Yval.astype(int)
	Yval = torch.from_numpy(Yval)

	class LambdaLayer(nn.Module):
		def __init__(self, lambd):
			super(LambdaLayer, self).__init__()
			self.lambd = lambd

		def forward(self, x):
			return self.lambd(x)

	Xtrain = Xtrain.to(torch.float32)
	Ytrain = Ytrain.to(torch.int64)
	Xval = Xval.to(torch.float32)
	Yval = Yval.to(torch.int64)

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

	class ResNetBlock(nn.Module):
		def __init__(self, numlayer, n):
			super(ResNetBlock, self).__init__()
			self.conv1 = nn.Conv2d(numlayer, numlayer, 3, padding = 1)
			self.batch_norm1 = Batch_Normalisation(numlayer,  0.1)
			self.conv2 = nn.Conv2d(numlayer, numlayer, 3, padding = 1)
			self.batch_norm2 = Batch_Normalisation(numlayer,  0.1)
		def forward(self, x):
			y = x
			x = self.conv1(x)
			x = self.batch_norm1(x)
			x = F.relu(x)
			x = self.conv2(x)
			x = self.batch_norm2(x)
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
			self.batch_norm1 = Batch_Normalisation(numlayerout,  0.1)
			self.conv2 = nn.Conv2d(numlayerout, numlayerout, 3, padding = 1)
			self.batch_norm2 = Batch_Normalisation(numlayerout,  0.1)
			self.s1A = LambdaLayer(lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, int(numlayerin/2), int(numlayerin/2)), "constant", 0))
		def forward(self, x):
			y = x
			x = self.conv1(x)
			x = self.batch_norm1(x)
			x = F.relu(x)
			x = self.conv2(x)
			x = self.batch_norm2(x)
			x = x + self.s1A(y)
			x = F.relu(x)
			return x

	class ResNet(nn.Module):
		def __init__(self, n1, r1):
			super(ResNet, self).__init__()
			self.n = n1
			self.r = r1
			self.conv_3_16 = nn.Conv2d(3, 16, 3, padding = 1)
			self.batch_norm1 = Batch_Normalisation(16,  0.1)
			self.resnet_layer1 = ResNet_Layer(16, n1)
			self.resnet_block1 = ResNet_Downsample(16, 32, n1)
			self.resnet_layer2 = ResNet_Layer(32, n1-1)
			self.resnet_block2 = ResNet_Downsample(32, 64, n1)
			self.resnet_layer3 = ResNet_Layer(64, n1-1)
			self.globalAvg = nn.AdaptiveAvgPool2d((1, 1))
			self.fc1 = nn.Linear(64, self.r)
		def forward(self, x):
			x =  self.conv_3_16(x)
			x = self.batch_norm1(x)
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
	model = model.to(device)

	batch_size = 128
	nepochs = 100
	train_data = TensorDataset(Xtrain, Ytrain)
	train_sampler = SequentialSampler(train_data)
	train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
	val_data = TensorDataset(Xval, Yval)
	val_sampler = SequentialSampler(val_data)
	val_dataloader = DataLoader(val_data, sampler = val_sampler, batch_size=batch_size)

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum = 0.9, weight_decay = 0.0001)
	scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 10, gamma = 0.1)

	train_loss = []
	def train(epoch_number):
		model.train()
		tloss = 0
		for s,b in enumerate(train_dataloader):
			b = [r.to(device) for r in b] 
			X, Y = b
			Y = Y.reshape(Y.shape[0])

			optimizer.zero_grad()        
			Yp, _ = model(X)

			loss = criterion(Yp, Y)
			loss.backward()
			tloss = tloss + loss.item()
			optimizer.step()
		scheduler.step()
		avg_loss = tloss / len(train_dataloader)
		train_loss.append(avg_loss)
		return avg_loss

	# function for evaluating the model
	val_loss = []
	# feature_quantile_1 = []
	# feature_quantile_20 = []
	# feature_quantile_80 = []
	# feature_quantile_99 = []
	def evaluate():  
		model.eval()
		total_loss, total_accuracy = 0, 0  
		correct = 0
		feature_all =np.array([])
		for step,batch in enumerate(val_dataloader): 
			batch = [r.to(device) for r in batch] 
			X, Y = batch
			Y = Y.reshape(Y.shape[0])
			with torch.no_grad():      
				preds, feature = model(X)
				feature_all = np.hstack((feature_all, feature.reshape(feature.shape[0]*feature.shape[1]).detach().cpu().numpy()))
				loss = criterion(preds, Y)
				total_loss = total_loss + loss.item()
				#print(preds.shape)
				#print(Y.shape)
				preds = preds.argmax(axis = 1)
				preds = preds.reshape(preds.shape[0])
				temp = ((preds) == Y)
				#print(preds.shape)
				#print(Y.shape)
				#print(temp)
				correct += temp.sum().float()
		# feature_quantile_1.append(np.percentile(feature_all, 1))
		# feature_quantile_20.append(np.percentile(feature_all, 20))
		# feature_quantile_80.append(np.percentile(feature_all, 80))
		# feature_quantile_99.append(np.percentile(feature_all, 99))
		avg_loss = total_loss / len(val_dataloader) 
		val_loss.append(avg_loss)
		accuracy = correct/(Xval.shape[0])
		return avg_loss, accuracy

	best_val_loss = float('inf')
	for epoch in range(nepochs):
		print('\n Epoch {:} / {:}'.format(epoch + 1, nepochs))
		tloss = train(epoch)
		vloss, valreport = evaluate()
		if vloss < best_val_loss:
			best_val_loss = vloss
			torch.save(model.state_dict(), output_file)
		print(f'\nTraining Loss: {tloss:.3f}')
		print(f'Validation Loss: {vloss:.3f}')
		print(f'Validation Report: {valreport}')

	