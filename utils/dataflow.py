import os 
import numpy as np

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from utils.dataset import *

def get_transforms(CONFIG):
    train_transform = transforms.Compose([
            transforms.RandomResizedCrop(CONFIG.input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CONFIG.mean, CONFIG.std)
        ])

    val_transform = transforms.Compose([
            transforms.RandomResizedCrop(CONFIG.input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CONFIG.mean, CONFIG.std)
        ])

    test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(CONFIG.mean, CONFIG.std)
        ])

    return train_transform, val_transform, test_transform




def get_dataset(train_transform, val_transform, test_transform, CONFIG):
    if CONFIG.datasets == "imagenet_lmdb":
        train_dataset, val_dataset, test_dataset = get_imagenet_lmdb(train_transform, val_transform, test_transform, CONFIG)

    elif CONFIG.datasets == "imagenet":
        train_dataset, val_dataset, test_dataset = get_imagenet(train_transform, val_transform, test_transform, CONFIG)

    else:
       raise

    return train_dataset, val_dataset, test_dataset


def get_dataloader(train_dataset, val_dataset, test_dataset, CONFIG):
    def _build_loader(dataset=None, shuffle=False, sampler=None):
        if dataset is not None:
            return torch.utils.data.DataLoader(
                        dataset,
                        batch_size=CONFIG.batch_size,
                        pin_memory=True,
                        num_workers=CONFIG.num_workers,
                    )
        return None

    train_loader = _build_loader(train_dataset, True)
    val_loader = _build_loader(val_dataset)
    test_loader = _build_loader(test_dataset)

    return train_loader, val_loader, test_loader


