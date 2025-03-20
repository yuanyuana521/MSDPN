import os
import torch
import numpy as np
from PIL import Image as Image
from data import PairCompose, PairRandomCrop, PairRandomHorizontalFilp, PairToTensor,PairResize
from torchvision.transforms import functional as F, transforms
from torch.utils.data import Dataset, DataLoader


def train_dataloader(path, batch_size=1, num_workers=0, use_transform=True):
    image_dir = os.path.join(path)

    transform = None
    if use_transform:
        transform = PairCompose(
            [

                PairRandomCrop((600,800)),
                PairRandomHorizontalFilp(),
                PairToTensor()
            ]
        )

    dataloader = DataLoader(
        DeblurDataset(image_dir, transform=transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader


def test_dataloader(path, batch_size=1, num_workers=0,use_transform=True):
    image_dir = os.path.join(path, 'ntire25_sh_rem_valid_inp')
    transform = None
    if use_transform:
        transform = PairCompose(
            [
                # PairResize((int(600), int(800))),
                # PairRandomCrop((480,360)),
                PairToTensor()
            ]
        )
    dataloader = DataLoader(
        DeblurDataset(image_dir, is_test=True, transform=transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader


def valid_dataloader(path, batch_size=1, num_workers=0,use_transform=True):
    transform = None
    if use_transform:
        transform = PairCompose(
            [
                PairResize((int(600), int(800))),
                # PairRandomCrop((480,360)),

                PairToTensor()
            ]
        )
    dataloader = DataLoader(
        DeblurDataset(os.path.join(path, 'test'),transform=transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return dataloader


class DeblurDataset(Dataset):
    def __init__(self, image_dir, transform=None, is_test=False):
        self.image_dir = image_dir
        self.image_list = os.listdir(os.path.join(image_dir, 'hazy/',))
        self._check_image(self.image_list)
        self.image_list.sort()
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_dir, 'hazy', self.image_list[idx]))
        # label = Image.open(os.path.join(self.image_dir, 'GT', self.image_list[idx].split('_')[0]+'_gt.png'))
        label = Image.open(os.path.join(self.image_dir, 'GT', self.image_list[idx].split('_')[0]))
        if self.transform:
            image, label = self.transform(image, label)
        else:
            image = F.to_tensor(image)
            label = F.to_tensor(label)
        if self.is_test:
            name = self.image_list[idx]
            return image, label, name
        return image, label

    @staticmethod
    def _check_image(lst):
        for x in lst:
            splits = x.split('.')
            if splits[-1] not in ['png', 'jpg', 'jpeg']:
                raise ValueError
