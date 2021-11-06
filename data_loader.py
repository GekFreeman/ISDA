from torchvision import datasets, transforms
import torch
import os
from torch.utils.data import Sampler
import pdb
def load_training(root_path, dir, batch_size, kwargs):
    transform = transforms.Compose(
        [transforms.Resize([256, 256]),
         transforms.RandomCrop(224),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor()])
    data = datasets.ImageFolder(root=os.path.join(root_path, dir), transform=transform)
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    return train_loader

def load_testing(root_path, dir, batch_size, kwargs):
    transform = transforms.Compose(
        [transforms.Resize([224, 224]),
         transforms.ToTensor()])
    data = datasets.ImageFolder(root=os.path.join(root_path, dir), transform=transform)
    test_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, **kwargs)
    return test_loader

def load_training_index(root_path, dir, batch_size, indices, kwargs):
    transform = transforms.Compose(
        [transforms.Resize([256, 256]),
         transforms.RandomCrop(224),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor()])
    sampler = SubsetSampler(indices)  
    data = datasets.ImageFolder(root=os.path.join(root_path, dir), transform=transform)
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, sampler=sampler, drop_last=True, **kwargs)
    return train_loader
    

class SubsetSampler(Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.
    Arguments:
        indices (sequence): a sequence of indices
    """
 
    def __init__(self, indices):
        self.indices = indices
 
    def __iter__(self):
        for i in range(len(self.indices)):
            yield self.indices[i]
 
    def __len__(self):
        return len(self.indices)