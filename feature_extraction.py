import os

import matplotlib.pyplot as plt
import pandas as pd
from dlup.data.dataset import TiledROIsSlideImageDataset
from dlup import SlideImage
from dlup.background import get_mask
import timm
import numpy as np
from tqdm import tqdm
import torch
from torchvision import transforms
import argparse
from PIL import Image, ImageDraw


class FeatureExtractor:
    def __init__(self, extraction_model, tile_size=(224, 224), target_mpp=2, normalize=False):
        self.model = extraction_model
        self.tile_size = tile_size
        self.target_mpp = target_mpp

        self.transforms = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                ]
            )

    def extract_features(self, slide_file, output_dir):
        """ Extract features for a given slide

        Parameters:
            slide_file: the slide to be processed
            output_dir: the dir where to store the encodings

        Returns:
            None: stores encodings and coords per WSI in the output_dir
        """
        # open the slide
        slide_image = SlideImage.from_file_path(slide_file)

        # generate the mask
        mask = get_mask(slide_image)

        # create a dataset for this WSI
        dataset = TiledROIsSlideImageDataset.from_standard_tiling(slide_file, self.target_mpp, self.tile_size, (0, 0),
                                                                  mask=mask)
        feature_list = []
        coord_list = []

        scaled_region_view = slide_image.get_scaled_view(slide_image.get_scaling(self.target_mpp))
        background = Image.new("RGBA", tuple(scaled_region_view.size), (255, 255, 255, 255))

        for d in dataset:
            # extract coordinates and features
            coords = np.array(d["coordinates"])
            tile = d['image'].convert('RGB')

            # to tensor, normalize and forward
            x = self.transforms(tile)
            x = torch.unsqueeze(x, 0)
            features = self.model(x).cpu().detach().numpy()

            # keep track
            feature_list.append(features)
            coord_list.append(coords)

            box = tuple(np.array((*coords, *(coords + self.tile_size))).astype(int))
            background.paste(tile, box)
            draw = ImageDraw.Draw(background)
            draw.rectangle(box, outline="red")

        # [S, H]
        feature_stack = np.stack(feature_list).squeeze()
        coord_stack = np.stack(coord_list)

        # save encodings + coordinates
        name = os.path.join(output_dir, slide_file.split('/')[-1][:-5])
        np.save(name + '_encoding', feature_stack)
        np.save(name + '_coords', coord_stack)

        # save plot
        plt.figure(figsize=(16, 10))
        plt.tight_layout()
        plt.imshow(background)
        plt.savefig(name + '_tiling.png')
        plt.cla()
        plt.close('all')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default='/data/archief/AMC-data/Barrett/LANS_001-923/')
    parser.add_argument("--output_dir", type=str,
                        default='/data/archief/AMC-data/Barrett/LANS_001-923/ImageNet_encoded/')
    parser.add_argument("--tile_size", type=tuple[int, int], default=(244, 244))
    parser.add_argument("--target_mpp", type=float, default=2.0)
    parser.add_argument("--model", type=str, default='resnet50')
    args = parser.parse_args()

    # define paths
    CASE_PATHS = [f for f in sorted(os.listdir(args.input_dir)) if 'RL' in f]

    # create model for feature extractor
    model = timm.create_model(args.model, pretrained=True, num_classes=0)
    feature_extractor = FeatureExtractor(model, tile_size=args.tile_size, target_mpp=args.target_mpp)
    print("Extracting {} encodings at {}".format(args.model, args.output_dir))
    print("Tile size: {}".format(args.tile_size))
    print("Target mpp: {}".format(args.target_mpp))

    blacklist = []
    for case in tqdm(CASE_PATHS):

        case_dir = os.path.join(args.output_dir, case)
        os.makedirs(case_dir, exist_ok=True)

        HE_files = sorted([os.path.join(args.input_dir, case, f) for f in
                           os.listdir(os.path.join(os.path.join(args.input_dir, case))) if 'HE' in f])
        for f in HE_files:

            try:
                feature_extractor.extract_features(f, output_dir=case_dir)
            except Exception:
                blacklist.append(f)

    print('Extraction failed for:\n')
    for f in blacklist:
        print(f)

    blacklist_df = pd.DataFrame({'cases': blacklist})
    blacklist_df.to_csv(os.path.join(args.output_dir, 'blacklist.csv'), index=False)
    print('Extraction done.')


