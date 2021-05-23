from cifar_train_resnet import *
from cifar_train_bin import *
from cifar_train_bn import *
from cifar_train_in import *
from cifar_train_gn import *
from cifar_train_ln import *
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
import copy
import argparse

# python3 train_cifar.py --normalization [ bn | in | bin | ln | gn | nn |
# torch_bn] --data_dir <directory_containing_data> --output_file <path to the
# trained model> --n [1 |  2 | 3 ] 
parser = argparse.ArgumentParser()
parser.add_argument("--normalization", dest = "normalization")
parser.add_argument("--data_dir",dest = "data_dir")
parser.add_argument("--output_file", dest ="output_file")
parser.add_argument("--n", dest = "n", type=int)
args = parser.parse_args()

normalization = args.normalization
data_dir = args.data_dir
output_file = args.output_file
n = args.n

# specify GPU/CPU
if torch.cuda.is_available():
	device = torch.device('cuda')
else:
	device = torch.device('cpu')

if normalization == "bn":
	run_full_code_bn(device, data_dir, output_file, n)
elif normalization == "in":
	run_full_code_in(device, data_dir, output_file, n)
elif normalization == "bin":
	run_full_code_bin(device, data_dir, output_file, n)
elif normalization == "ln":
	run_full_code_ln(device, data_dir, output_file, n)
elif normalization == "gn":
	run_full_code_gn(device, data_dir, output_file, n)
else:
	run_full_code_resnet(device, data_dir, output_file, n)

