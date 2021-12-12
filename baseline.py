import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import argparse
import json
import logging
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

# # # MODELS # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)
# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
    
# ResNet
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[1], 2)
        self.layer3 = self.make_layer(block, 64, layers[2], 2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
            
# # # LIBRARY # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def t_avg(model, transform, images, classes, device):
    outputs = torch.zeros(len(images), classes).to(device)
    for i in range(30):
        outputs += model(transform(images).to(device))
    outputs /= 30
    return outputs

def accuracy_test(model, loader, transform, apply_transform, device):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            if (apply_transform):
                outputs = model(transform(images))
            else:
                outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        model.train()
        return (100.0000 * correct / total)
        
def t_avg_test(model, loader, transform, classes, device):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = t_avg(model, transform, images, classes, device)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        model.train()
        return (100.0000 * correct / total)
        
def double_test(model, loader, transform, device):
    results = [accuracy_test(model, loader, transform, False, device), 
               accuracy_test(model, loader, transform, True, device)]
    return results

def triple_test(model, loader, transform, classes, device):
    results = [accuracy_test(model, loader, transform, False, device),
               accuracy_test(model, loader, transform, True, device),
               t_avg_test(model, loader, transform, classes, device)]
    return results
    
# # # DATASET # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #    

class CIFAR10:
    
    def __init__(self, batch):
        self.batch = batch
        self.stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'), 
            transforms.RandomHorizontalFlip(), 
            transforms.ToTensor(), 
            transforms.Normalize(*self.stats,inplace=True)])
        transform_train_aug = transforms.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'), 
            transforms.RandomCrop(24),
            transforms.Resize(32),
            transforms.RandomHorizontalFlip(), 
            transforms.ToTensor(), 
            transforms.Normalize(*self.stats,inplace=True)])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*self.stats)])
        
        train = torchvision.datasets.CIFAR10(root='../../data/', train=True, transform=transform_train, download=True)
        train_aug = torchvision.datasets.CIFAR10(root='../../data/', train=True, transform=transform_train_aug, download=True)
        test = torchvision.datasets.CIFAR10(root='../../data/', train=False, transform=transform_test)
        
        self.train_loader = torch.utils.data.DataLoader(dataset=train, batch_size=batch, shuffle=True, num_workers=2)
        self.train_aug_loader = torch.utils.data.DataLoader(dataset=train_aug, batch_size=batch, shuffle=True, num_workers=2)
        self.test_loader = torch.utils.data.DataLoader(dataset=test, batch_size=100, shuffle=True, num_workers=2)
        
# # # MAIN # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    
def update_lr(optimizer, lr):    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
def load_data(args):
    if args.dataset == 'cifar10':
        return CIFAR10(args.batch_size)
    
def load_arch(args):
    if args.arch == 'resnet18':
        return ResNet(ResidualBlock, [2, 2, 2])
        
def train(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    augment = torch.nn.Sequential(
        transforms.RandomCrop(24),
        transforms.Resize(32),
        transforms.RandomHorizontalFlip())
    
    ds = load_data(args)
    train_loader = ds.train_loader
    test_loader = ds.test_loader
    
    model = load_arch(args).to(device)
    e = args.epochs
    curr_lr = args.lr
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    for epoch in range(e):
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            images, labels = images.to(device), labels.to(device)
            loss = nn.CrossEntropyLoss()(model(images), labels)
            loss.backward()
            optimizer.step()

        if ( (epoch+1) == 100 ) or ( (epoch+1) == 150 ):
            curr_lr *= 0.1
            update_lr(optimizer, curr_lr)
        if (epoch+1) % args.loss_interval == 0:
            logger.info( "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format( epoch+1, e, i+1, len(train_loader), loss.item() ) )
        if (epoch+1) % args.test_interval == 0:
            results = double_test(model, test_loader, augment, device)
            logger.info( "Student: UA {:.4f}, A {:.4f}".format( results[0], results[1] ) )
    
    save_model(model, args.model_dir)

def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_arch(args)
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f))
    return model.to(device)

def save_model(model, model_dir):
    with open(os.path.join(model_dir, 'model.pth'), 'wb') as f:
        torch.save(model.state_dict(), f)
    logger.info("model saved")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--arch', type=str, default='resnet18', choices=['resnet18'])
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10'], help='...')
    parser.add_argument("--batch_size", type=int, default=256, metavar="N", help="training batch size")
    parser.add_argument('--epochs', type=int, default=200, metavar="N", help='number of training epochs')
    parser.add_argument('--lr', type=float, default=0.1, metavar="LR", help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, metavar="M", help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=5e-4, metavar="W", help='weight decay')
    parser.add_argument('--loss_interval', type=int, default=5, metavar="N", help='loss printing interval')
    parser.add_argument('--test_interval', type=int, default=25, metavar="N", help='test run interval')
    
    # Container environment
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--current_host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data_dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument("--num_gpus", type=int, default=os.environ["SM_NUM_GPUS"])
    
    train(parser.parse_args())
