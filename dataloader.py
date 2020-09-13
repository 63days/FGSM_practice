import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from torch.utils.data import DataLoader

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


class DataSetWrapper(object):
    def __init__(self, batch_size, num_workers, valid_size):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def get_train_validation_data_loaders(self):
        ds = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=self.transform)

        indices = list(range(len(ds)))
        np.random.shuffle(indices)

        split = int(np.floor(self.valid_size * len(ds)))
        train_idx, valid_idx = indices[split:], indices[:split]

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = DataLoader(ds, batch_size=self.batch_size, sampler=train_sampler,
                                  num_workers=self.num_workers, shuffle=False)
        valid_loader = DataLoader(ds, batch_size=self.batch_size, sampler=valid_sampler,
                                  num_workers=self.num_workers, shuffle=False)

        return train_loader, valid_loader

    def get_test_data_loader(self):
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=self.transform)
        test_loader = DataLoader(testset, batch_size=self.batch_size,
                                                 shuffle=False, num_workers=self.num_workers)

        return test_loader

    def get_train_data_loader(self):
        ds = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=self.transform)
        train_loader = DataLoader(ds, batch_size=self.batch_size,
                                  num_workers=self.num_workers, shuffle=False)
        return train_loader