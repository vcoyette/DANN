"""Utils classes and functions."""
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
