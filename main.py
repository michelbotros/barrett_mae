import torch
from vit_pytorch import ViT, MAE
from dataloader import BarrettsTissue
from torch.utils.data import DataLoader
from torch.optim import AdamW
from dataloader import get_wsi_paths
import argparse
from tqdm import tqdm


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)

    # training parameters
    parser.add_argument('--batch_size', default=256, type=int,
                        help='batch size')
    parser.add_argument('--epochs', default=400, type=int,
                        help='number of epochs to pre-train')

    # encoder model parameters
    parser.add_argument('--input_size', default=512, type=int,
                        help='images input size')
    parser.add_argument('--target_mpp', default=2, type=int,
                        help='resolution of the tiles')
    parser.add_argument('--patch_size', default=32, type=int,
                        help='patch size for masking')
    parser.add_argument('--num_classes', default=3, type=int,
                        help='number of classes (not used during pretraining)')
    parser.add_argument('--encoder_dim', default=1024, type=int,
                        help='')
    parser.add_argument('--encoder_depth', default=6, type=int,
                        help='')
    parser.add_argument('--encoder_heads', default=8, type=int,
                        help='')
    parser.add_argument('--encoder_mlp_dim', default=2048, type=int,
                        help='')

    # MAE model parameters
    parser.add_argument('--masking_ratio', default=0.75, type=float,
                            help='masking ratio (percentage of removed patches)')
    parser.add_argument('--decoder_dim', default=512, type=int,
                        help='')
    parser.add_argument('--decoder_depth', default=6, type=int,
                        help='')

    # optimizer (AdamW) parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=1e-5, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--betas', type=float, default=(0.9, 0.95),
                        help='betas in AdamW')

    # scheduler parameters TODO (not used yet)
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # dataset parameters TODO (not used yet)
    parser.add_argument('--data_paths', default='split.yml', type=str,
                        help='dataset path')
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')

    # device parameters
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')

    return parser


def pretrain(args):
    """
    Pre-train a MAE model for Barrett's tissue.

    Parameters:

    """
    # set device
    device = torch.device(args.device)

    # get paths to training files
    tiff_files = get_wsi_paths(args.data_paths, partition='training')[:5]

    # create a datasets of Barrett's tissue patches
    dataset = BarrettsTissue(tiff_files=tiff_files, tile_size=(args.input_size, args.input_size), target_mpp=args.target_mpp)
    print('Dataset consists of {} WSIs with {} tiles'.format(len(tiff_files), len(dataset)))
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # construct encoder model
    v = ViT(
        image_size=args.input_size,
        patch_size=args.patch_size,
        num_classes=args.num_classes,
        dim=args.encoder_dim,
        depth=args.encoder_depth,
        heads=args.encoder_heads,
        mlp_dim=args.encoder_mlp_dim
    )

    # construct MAE model
    mae = MAE(
        encoder=v,
        masking_ratio=args.masking_ratio,       # the paper recommended 75% masked patches
        decoder_dim=args.decoder_dim,           # paper showed good results with just 512
        decoder_depth=args.decoder_depth        # anywhere from 1 to 8
    ).to(device)

    # define optimizer
    optimizer = AdamW(params=mae.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=args.betas)

    # pre-train
    for e in tqdm(range(args.epochs), desc='Epoch'):
        epoch_loss = 0.0
        for image_batch in tqdm(train_dataloader, desc='Batch', leave=False):
            loss = mae(image_batch.to(device))
            loss.backward()
            optimizer.step()
            epoch_loss += loss
        print('Epoch {}, loss: {:.3f}'.format(e, epoch_loss / len(train_dataloader)))

    # save the pre-trained encoder
    torch.save(v.state_dict(), './trained-encoder.pt')


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    pretrain(args)

