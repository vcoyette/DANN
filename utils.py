"""Utils classes and functions."""
import os

from PIL import Image
import torch


class CustomDataset(torch.utils.data.Dataset):
    """Custom Dataset."""

    def __init__(self, data_root, data_list, transform=None):
        """Initialize CustomDataset.

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
