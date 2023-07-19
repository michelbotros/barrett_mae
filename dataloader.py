import os
from dlup.data.dataset import TiledROIsSlideImageDataset
from dlup import SlideImage
from dlup.background import get_mask
from torch.utils.data import Dataset, DataLoader
import random
from torchvision.transforms import ToTensor, RandomRotation, RandomResizedCrop, RandomHorizontalFlip, Compose, RandomApply, RandomVerticalFlip
import yaml
from tqdm import tqdm


class BarrettsTissue(Dataset):

    def __init__(self, tiff_files, tile_size=(256, 256), target_mpp=2, mask_threshold=0.3):
        self.files = tiff_files
        self.tile_size = tile_size
        self.target_mpp = target_mpp

        self.datasets = []
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

        transforms_train = Compose([
            RandomApply([RandomRotation(90, 90)], p=0.5),
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            ToTensor()
        ])
        return transforms_train(tile)


def get_wsi_paths(split_file, partition='training'):
    with open(split_file, 'r') as file:
        split = yaml.safe_load(file)
    paths = [x['wsi']['path'] for x in split[partition]]
    return paths
