"""colored mnist, rescaled to 16x16"""

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T 
from torchvision.datasets import MNIST
import numpy as np


DEFAULT_DATASET_PATH = "./data"


def random_color(im):
    hue = 360*np.random.rand()
    d = (im *(hue%60)/60)
    im_min, im_inc, im_dec = torch.zeros_like(im), d, im - d
    H = round(hue/60) % 6    
    cmap = [[0, 3, 2], [2, 0, 3], [1, 0, 3], [1, 2, 0], [3, 1, 0], [0, 1, 2]]
    return torch.cat((im, im_min, im_dec, im_inc), dim=0)[cmap[H]]


class CMNISTDataset(Dataset):
    def __init__(
        self,
        train=True,
        spat_dim=(16, 16),
        root=DEFAULT_DATASET_PATH,
        download=True,
        pix_range=(0., 1.)
    ) -> None:
        super().__init__()
        _m, _std = pix_range[0]/float(pix_range[0] - pix_range[1]), 1./float(pix_range[1] - pix_range[0])
        TRANSFORM = T.Compose([
            T.Resize(spat_dim),
            T.ToTensor(),
            random_color,
            T.Normalize([_m],[_std])
        ])
        self.mnist_digit = MNIST(root=root, train=train, download=download, transform=TRANSFORM)

    def __len__(self) -> int:
        return len(self.mnist_digit)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.mnist_digit[idx][0]

        
