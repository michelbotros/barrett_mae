import torch
import numpy as np
import os
from vit_pytorch import ViT, MAE
from dataloader import BarrettsTissue
from torch.utils.data import DataLoader
from torch.optim import AdamW
from dataloader import get_wsi_paths


def pretrain():
    """
    Pre-train a MAE model.

    Parameters:

    """
    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # get paths to training files
    tiff_files = get_wsi_paths('split.yml', partition='training')[:100]

    # create a datasets of Barrett's tissue patches
    dataset = BarrettsTissue(tiff_files=tiff_files, tile_size=(256, 256), target_mpp=2)
    print('Dataset consists of {} WSIs with {} tiles'.format(len(tiff_files), len(dataset)))
    train_dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)

    # construct encoder model
    v = ViT(
        image_size=256,
        patch_size=32,
        num_classes=3,
        dim=1024,
        depth=6,
        heads=8,
        mlp_dim=2048
    )

    # construct MAE model
    mae = MAE(
        encoder=v,
        masking_ratio=0.75,    # the paper recommended 75% masked patches
        decoder_dim=512,       # paper showed good results with just 512
        decoder_depth=6        # anywhere from 1 to 8
    ).to(device)

    # define training setup
    optimizer = AdamW(params=mae.parameters(), lr=1e-5, weight_decay=0.05, betas=(0.9, 0.95))
    epochs = 100

    # pre-train
    for e in range(epochs):
        epoch_loss = 0.0

        for image_batch in train_dataloader:
            loss = mae(image_batch.to(device))
            loss.backward()
            optimizer.step()
            epoch_loss += loss

        print('Epoch {}, loss: {:.3f}'.format(e, epoch_loss / len(train_dataloader)))

    # save the pre-trained encoder
    torch.save(v.state_dict(), './trained-encoder.pt')


if __name__ == '__main__':
    # args
    pretrain()

