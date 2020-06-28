import argparse
import os

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
from copy import copy

from net.models import LeNet
import util

os.makedirs('saves', exist_ok=True)

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST pruning from deep compression paper')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 100)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=100, metavar='N', #init 250
                    help='number of epochs to train (default: 100)')
                    
parser.add_argument('--reg_type', type=int, default=0, metavar='R',
                    help='regularization type: 0:None 1:L1 2:Hoyer 3:HS 4:Transformed L1')
parser.add_argument('--decay', type=float, default=0.001, metavar='D',
                    help='weight decay for regularizer (default: 0.001)')

parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=12345678, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--log', type=str, default='log.txt',
                    help='log file name')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
args = parser.parse_args()

# Control Seed
torch.manual_seed(args.seed)

# Select Device
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else 'cpu')
if use_cuda:
    print("Using CUDA!")
    torch.cuda.manual_seed(args.seed)
else:
    print('Not using CUDA!!!')

# Loader
kwargs = {'num_workers': 5, 'pin_memory': True} if use_cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor()])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, transform=transforms.Compose([
                       transforms.ToTensor()])),
    batch_size=args.test_batch_size, shuffle=False, **kwargs)


# Define which model to use
model = LeNet(mask=False).to(device)
#print(model)
#print(args)
util.print_model_parameters(model)

# NOTE : `weight_decay` term denotes L2 regularization loss term
optimizer = optim.Adam(model.parameters(), lr=args.lr)
initial_optimizer_state_dict = optimizer.state_dict()

def train(epochs, decay=0, threshold=0.0):
    model.train()
    pbar = tqdm(range(epochs), total=epochs)
    curves = np.zeros((epochs,14))
    
    for epoch in pbar:
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            
            reg = 0.0
            if decay:
                reg = 0.0
                for param in model.parameters():
                    if param.requires_grad and torch.sum(torch.abs(param))>0:
                        # l1 norm minimizaition
                        if args.reg_type==1:
                            reg += torch.sum(torch.abs(param))
                        elif args.reg_type==2:                        # l1/l2 norm minimization
                            reg += torch.sum(torch.abs(param))/torch.sqrt(torch.sum(param**2))
                        elif args.reg_type==3:                        # (l1/l2)**2 minimization
                            reg += (torch.sum(torch.abs(param))**2)/torch.sum(param**2)
                        elif args.reg_type==4:         # (2*l1)/(1+l1)
                            reg += torch.sum(2*torch.abs(param)/(1+torch.abs(param)))
                        elif args.reg_type==5:         # reweighted l2 -> lp
                            param_ = param.clone().detach().requires_grad_(False)
                            eps = 1e-10
                            p = 1/2.0
                            reg += torch.sum(torch.pow(torch.abs(param),2)  * torch.pow(torch.abs(param_) + eps, (p-2.0)/2.0))
                        elif args.reg_type==6:         # reweighted l1 -> lp
                            param_ = param.clone().detach().requires_grad_(False)
                            eps = 1e-10
                            p = 1/2.0
                            reg += torch.sum(torch.abs(param)  * torch.pow(torch.abs(param_) + eps, p-1))
                        #https://www.ece.uvic.ca/~andreas/JournalPapers/JKP-WSL-AA,Improved_Compessive_Sensing_Algorithms,TCASII14.pdf
                        else:
                            reg = 0.0         
            total_loss = loss+decay*reg
            total_loss.backward()
            optimizer.step()
            
            if batch_idx % args.log_interval == 0:
                done = batch_idx * len(data)
                percentage = 100. * batch_idx / len(train_loader)
                pbar.set_description(f'Train Epoch: {epoch} [{done:5}/{len(train_loader.dataset)} ({percentage:3.0f}%)]  Loss: {loss.item():.3f}  Reg: {reg:.3f}')


def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
        print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
    return accuracy

if args.pretrained:
    model.load_state_dict(torch.load('saves/Elt_Decay_0.0_Regtype_0_Epoch_250.pth'))
    accuracy = test()

# Initial training
print("--- Beging  Training ---\n")
util.log(args.log, f"--- Beging  Training ---")

train(args.epochs, decay=args.decay, threshold=0.0)
accuracy = test()
model_name = 'saves/Elt_Decay_'+str(args.decay)+'_Regtype_'+str(args.reg_type)+'_Epoch_'+str(args.epochs)+'.pth'
with open('./model_name.txt', 'w') as fp:
    fp.write(model_name)
torch.save(model.state_dict(), model_name)
util.log(args.log, f"model saved path: {model_name}")
util.log(args.log, f"accuracy {accuracy}")
print(args.log, f"accuracy {accuracy}")
print("--- Finish  Training ---\n")
util.log(args.log, f"---  Finish  Training ---")


#util.print_nonzeros(model)


