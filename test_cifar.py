from cifar_test_resnet import *
from cifar_test_bin import *
from cifar_test_bn import *
from cifar_test_in import *
from cifar_test_gn import *
from cifar_test_ln import *
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

# python3 test_cifar.py -model_file <path to the trained model> --normalization
# [ bn | in | bin | ln | gn | nn | inbuilt ] --n [ 1 |  2 | 3  ]
# --test_data_file <path to a csv with each line representing an image>
# --output_file <file containing the prediction in the same order as in the
# input csv>
parser = argparse.ArgumentParser()
parser.add_argument("--normalization", dest = "normalization")
parser.add_argument("--model_file",dest = "model_file")
parser.add_argument("--output_file", dest ="output_file")
parser.add_argument("--test_data_file", dest ="test_data_file")
parser.add_argument("--n", dest = "n", type=int)
args = parser.parse_args()

model_file = args.model_file
normalization = args.normalization
test_data_file = args.test_data_file
output_file = args.output_file
n = args.n

# specify GPU/CPU
if torch.cuda.is_available():
	device = torch.device('cuda')
else:
	device = torch.device('cpu')

if normalization == "bn":
	run_full_code_bn(device, model_file, test_data_file, output_file, n)
elif normalization == "in":
	run_full_code_in(device, model_file, test_data_file, output_file, n)
elif normalization == "bin":
	run_full_code_bin(device, model_file, test_data_file, output_file, n)
elif normalization == "ln":
	run_full_code_ln(device, model_file, test_data_file, output_file, n)
elif normalization == "gn":
	run_full_code_gn(device, model_file, test_data_file, output_file, n)
else:
	run_full_code_resnet(device, model_file, test_data_file, output_file, n)

