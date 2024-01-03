import os
from dlup.data.dataset import TiledROIsSlideImageDataset
from dlup import SlideImage
from dlup.background import get_mask
from torch.utils.data import Dataset, DataLoader
import random
from torchvision.transforms import ToTensor, RandomRotation, RandomResizedCrop, RandomHorizontalFlip, Compose, RandomApply, RandomVerticalFlip
import yaml
from tqdm import tqdm
import numpy as np
import torch
import pandas as pd


class SlideDataset(Dataset):
    def __init__(self, x, y, ids=None):
        """
        Parameters:
            x: list of numpy arrays (arbi)
            y:
            ids:
        """
        self.slide_sequences = x
        self.slide_labels = y
        self.ids = ids

    def __len__(self):
        return len(self.slide_sequences)

    def get_slide_id(self, idx):
        return self.ids[idx]

    def __getitem__(self, idx):
        x = torch.Tensor(self.slide_sequences[idx])              # (L, H_in)
        y = torch.Tensor(self.slide_labels[idx])                 # (1)
        return x, y


class BarrettsTissueMAE(Dataset):
    """Dataset used for training a MAE model. Draws """

    def __init__(self, tiff_files, tile_size=(256, 256), target_mpp=2, mask_threshold=0.3, transforms=True):
        self.files = tiff_files
        self.tile_size = tile_size
        self.target_mpp = target_mpp
        self.datasets = []
        self.transforms = transforms

        for f in tqdm(tiff_files, desc='Pre-loading WSIs...'):
            slide_image = SlideImage.from_file_path(f)

            # generate the mask
            mask = get_mask(slide_image)

            # create a dataset for this WSI
            dataset = TiledROIsSlideImageDataset.from_standard_tiling(f, self.target_mpp, self.tile_size, (0, 0),
                                                                      mask=mask, mask_threshold=mask_threshold)
            self.datasets.append(dataset)

    def __len__(self):
        """
        Number of tiles
        """
        return sum([len(d) for d in self.datasets])

    def __getitem__(self, item):

        # get a random tile from a random WSI
        wsi = random.choice(self.datasets)
        tile = random.choice(wsi)['image'].convert('RGB')

        # define standard transforms
        transforms_train = Compose([
            RandomApply([RandomRotation((90, 90))], p=0.5),
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            ToTensor()
        ])

        if self.transforms:
            tile = transforms_train(tile)

        return tile


def get_wsi_paths(split_file, partition='training'):
    with open(split_file, 'r') as file:
        split = yaml.safe_load(file)
    paths = [x['wsi']['path'] for x in split[partition]]

    return paths
