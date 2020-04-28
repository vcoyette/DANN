"""Utility functions for data."""
import os

import torch
import torchvision
import torchvision.transforms as transforms

from data.dataset import MNIST_M


def load_source(**kwargs):
    """Load source dataloader (MNIST).

    :**kwargs: arguments to pass to dataloader constructor
    :returns: train_loader, test_loader

    """
    # Load source images
    transform_source = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,)),
         # Make 3D images from grayscal MNIST
         transforms.Lambda(lambda x: x.repeat(3, 1, 1))])

    trainset_source = torchvision.datasets.MNIST(root='./data',
                                                 train=True,
                                                 download=True,
                                                 transform=transform_source)

    testset_source = torchvision.datasets.MNIST(root='./data',
                                                train=False,
                                                download=True,
                                                transform=transform_source)

    return get_loader(trainset_source, **kwargs),\
        get_loader(testset_source, **kwargs)


def load_target(**kwargs):
    """Load target dataloader (MNIST_M).

    :**kwargs: arguments to pass to dataloader constructor
    :returns: train_loader, test_loader

    """
    # Load target images
    img_transform_target = transforms.Compose([
        transforms.Resize(28),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    trainset_target = MNIST_M(
        data_root=os.path.join('data', 'mnist_m', 'mnist_m_train'),
        data_list=os.path.join('data', 'mnist_m', 'mnist_m_train_labels.txt'),
        transform=img_transform_target
    )

    testset_target = MNIST_M(
        data_root=os.path.join('data', 'mnist_m', 'mnist_m_test'),
        data_list=os.path.join('data', 'mnist_m', 'mnist_m_test_labels.txt'),
        transform=img_transform_target
    )
    return get_loader(trainset_target, **kwargs),\
        get_loader(testset_target, **kwargs)


def get_loader(dataset, **kwargs):
    """Get dataloader from dataset."""
    return torch.utils.data.DataLoader(dataset, **kwargs)
