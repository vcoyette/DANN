"""Utility functions for data."""
import os

import torch
import torchvision
import torchvision.transforms as transforms

from data.dataset import MNIST_M


def load_mnist(**kwargs):
    """Load MNIST dataloader, repeating along 3 channels.

    :**kwargs: arguments to pass to dataloader constructor
    :returns: train_loader, test_loader

    """
    # Load source images
    transform_source = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,)),
         # Make 3D images from grayscal MNIST
         transforms.Lambda(lambda x: x.repeat(3, 1, 1))])

    trainset = torchvision.datasets.MNIST(root='./data',
                                          train=True,
                                          download=True,
                                          transform=transform_source)

    testset = torchvision.datasets.MNIST(root='./data',
                                         train=False,
                                         download=True,
                                         transform=transform_source)

    return get_loader(trainset, **kwargs),\
        get_loader(testset, **kwargs)


def load_svhn(**kwargs):
    """Load SVHM dataloader.

    :**kwargs: arguments to pass to dataloader constructor
    :returns: train_loader, test_loader

    """
    # Load source images
    transform_source = transforms.Compose(
        [transforms.Resize(28),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

    trainset = torchvision.datasets.SVHN(root='./data',
                                         split='train',
                                         download=True,
                                         transform=transform_source)

    testset = torchvision.datasets.SVHN(root='./data',
                                        split='test',
                                        download=True,
                                        transform=transform_source)

    return get_loader(trainset, **kwargs),\
        get_loader(testset, **kwargs)


def load_mnist_m(**kwargs):
    """Load MNIST_M dataloader.

    :**kwargs: arguments to pass to dataloader constructor
    :returns: train_loader, test_loader

    """
    # Load target images
    img_transform = transforms.Compose([
        transforms.Resize(28),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    trainset = MNIST_M(
        data_root=os.path.join('data', 'mnist_m', 'mnist_m_train'),
        data_list=os.path.join('data', 'mnist_m', 'mnist_m_train_labels.txt'),
        transform=img_transform
    )

    testset = MNIST_M(
        data_root=os.path.join('data', 'mnist_m', 'mnist_m_test'),
        data_list=os.path.join('data', 'mnist_m', 'mnist_m_test_labels.txt'),
        transform=img_transform
    )
    return get_loader(trainset, **kwargs),\
        get_loader(testset, **kwargs)


def get_loader(dataset, **kwargs):
    """Get dataloader from dataset."""
    return torch.utils.data.DataLoader(dataset, **kwargs)
