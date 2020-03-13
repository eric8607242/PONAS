import os

import torch
import torchvision.datasets as datasets
from torch.utils.data.sampler import SubsetRandomSampler

from utils.dataset.folder2lmdb import ImageFolderLMDB


def get_imagenet_lmdb(train_transform, val_transform, test_transform, CONFIG):
    """
    Load lmdb imagenet dataset
    https://github.com/Fangyh09/Image2LMDB
    """
    def _build_dataset(data_path, transform=None):
        if os.path.isfile(data_path) and transform:
            return ImageFolderLMDB(data_path, transform, None)

    train_path = os.path.join(CONFIG.dataset_dir, "train_lmdb", "train.lmdb")
    val_path = os.path.join(CONFIG.dataset_dir, "val_lmdb", "val.lmdb")
    test_path = os.path.join(CONFIG.dataset_dir, "test_lmdb", "test.lmdb")

    train_data = _build_dataset(train_path, train_transform)
    val_data = _build_dataset(val_path, val_transform)
    test_data = _build_dataset(test_path, test_transform)

    return train_data, val_data, test_data

def get_imagenet(train_transform, val_transform, test_transform, CONFIG):
    def _build_dataset(data_path, transform=None):
        if os.path.isdir(data_path) and transform is not None:
            return datasets.ImageFolder(data_path, transform)

    train_path = os.path.join(CONFIG.dataset_dir, "train")
    val_path = os.path.join(CONFIG.dataset_dir, "val")
    test_path = os.path.join(CONFIG.dataset_dir, "test")

    train_data = _build_dataset(train_path, train_transform)
    val_data = _build_dataset(val_path, val_transform)
    test_data = _build_dataset(test_path, test_transform)

    return train_data, val_data, test_data


    

