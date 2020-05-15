"""Utils classes and functions."""
import cv2
import glob
import os

from PIL import Image
from torch.utils.data import Dataset


class MNIST_M(Dataset):
    """MNIST_M Dataset."""

    def __init__(self, data_root, data_list, transform=None):
        """Initialize MNIST_M data set.

        Keyword Params:
            root -- the root folder containing the data
            data_list -- the file containing list of images - labels
            transform (optional) -- tranfrom images when loading
        """
        self.root = data_root
        self.transform = transform
        with open(data_list, 'r') as f:
            self.data_list = f.readlines()

    def __len__(self):
        """Get length of dataset."""
        return len(self.data_list)

    def __getitem__(self, idx):
        """Get item."""
        img_name, labels = self.data_list[idx].split()
        imgs = Image.open(os.path.join(self.root, img_name)).convert('RGB')

        if self.transform:
            imgs = self.transform(imgs)

        labels = int(labels)

        return imgs, labels


class Office(Dataset):
    """Office Dataset."""

    def __init__(self, data_root, dataset, transform=None):
        """Initialize an Office dataset.

        Keyword Parameters:
            data_root -- root of the office dataset
            dataset -- name of the dataset to use,
                       one of ['amazon', 'dslr', 'webcam']
            trainsform (optional) -- transform images when loading
        """
        # Validate dataset parameter
        available_datasets = ['amazon', 'dslr', 'webcam']
        if dataset.lower() not in available_datasets:
            raise ValueError(f'Unknown dataset: {dataset}.'
                             f'Chose one of: {available_datasets}')

        # Get dataset directory
        dataset_dir = os.path.join(data_root, dataset, 'images')

        # Images path
        img_regexp = os.path.join(dataset_dir, '*/*')
        self.images = glob.glob(img_regexp)

        # Get labels
        classes = sorted(os.listdir(dataset_dir))
        self.labels = list(map(lambda img: classes.index(img.split('/')[-2]),
                               self.images))

        self.transform = transform

    def __len__(self):
        """Get length of dataset."""
        return len(self.images)

    def __getitem__(self, idx):
        """Get item."""
        img_name, label = self.images[idx], self.labels[idx]

        img = cv2.imread(img_name)
        img = cv2.resize(img, (256, 256))

        if self.transform:
            img = self.transform(img)

        return img, label
