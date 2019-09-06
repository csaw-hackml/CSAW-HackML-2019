'''Test Backdoor Network with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

from architecture import *
import h5py

# create the dataloader
class Data(Dataset):

    def __init__(self, h5_path1, transform=None):
        self.transform = transform
        self.h51 = h5_path1
        f1 = h5py.File(h5_path1,'r')
        self.length = len(f1['label'])

    # Override to give PyTorch access to any image on the dataset.
    def __getitem__(self, index):
        if index < self.length:
            with h5py.File(self.h51, 'r') as h5f:
                if self.transform is not None:
                    X = self.transform(h5f['data'][index].astype('uint8'))
                else:
                    X = torch.from_numpy(h5f['data'][index].transpose(2, 0, 1))/255.0
                y = h5f['label'][index].astype('long')
        return X, y
    # Override to give PyTorch size of dataset
    def __len__(self):
        return self.length

parser = argparse.ArgumentParser(description='PyTorch Face Recognition Testing')
parser.add_argument('--model', type=str, required=True, help='specify the model path')
parser.add_argument('--seed', type=int, default=1)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy

torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Data
print('==> Preparing data..')

transform_test = transforms.Compose([
    # transforms.ToPILImage(),
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


dset_test = Data('./data/clean_data/test.h5',transform=transform_test)
kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
testloader = torch.utils.data.DataLoader(dset_test, batch_size=10, shuffle=False, **kwargs)

# Model
print('==> Building model..')
net = densenet_BC_Face(100, 12, num_classes=1284)
net = net.to(device)


# Load checkpoint.
print('==> Resuming from checkpoint..')
# assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
net.load_state_dict(torch.load(args.model))

if device == 'cuda':
    #net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

def test():
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    acc = 100.*correct/total
    if acc > best_acc:
        print('The classification accuracy is {:.4f}'.format(acc))
        best_acc = acc
if __name__ == '__main__':
    test()

