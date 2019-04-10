import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image
from torchvision.datasets.folder import pil_loader
import vltools.image as vlimage
import os, sys
from os.path import join, split, splitext, abspath, dirname, isfile, isdir

def ilsvrc2012(path, bs=256, num_workers=8):
    traindir = os.path.join(path, 'train')
    valdir = os.path.join(path, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=bs, shuffle=True,
        num_workers=num_workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=bs, shuffle=False,
        num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader

def cifar10(path='data/cifar10', bs=100, num_workers=8):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_dataset = datasets.CIFAR10(root=path, train=True, download=True,
                                                 transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True,
                                               num_workers=num_workers)

    test_dataset = datasets.CIFAR10(root=path, train=False, download=True,
                                                transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False,
                                               num_workers=num_workers)

    return train_loader, test_loader

def cifar100(path='data/cifar100', bs=256, num_workers=8):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_dataset = datasets.CIFAR100(root=path, train=True, download=True,
                                                 transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True,
                                               num_workers=num_workers)

    test_dataset = datasets.CIFAR100(root=path, train=False, download=True,
                                                transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, 
                                               num_workers=num_workers)

    return train_loader, test_loader

class CACDDataset(torch.utils.data.Dataset):

    def __init__(self, root, filelist):

        self.root = root
        assert isfile(filelist)

        # list files: `cacd_train_list.txt`, `cacd_test_list.txt`, `cacd_val_list.txt`
        with open(filelist) as f:
            self.items = f.readlines()

    def __getitem__(self, index):

        filename, age = self.items[index]
        age = int(age)
        im = pil_loader(join(self.root, filename))

        return im, age

    def __len__(self):
        return len(self.items)

class LFWDataset(torch.utils.data.Dataset):

    def __init__(self, root, filelist):

        self.root = root
        assert isfile(filelist)

        with open(filelist) as f:
            self.items = f.readlines()

        for i in range(self.__len__()):
            self.items[i] = self.items[i].strip()

    def __getitem__(self, index):

        filename = self.items[index]
        assert isfile(join(root, filename))
        im = pil_loader(join(self.root, filename))

        return im

    def __len__(self):
        return len(self.items)
