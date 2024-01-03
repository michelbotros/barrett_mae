import torch
from vit_pytorch import ViT, MAE
import lr_sched
from dataloader import BarrettsTissueMAE
from torch.utils.data import DataLoader
from torch.optim import AdamW
from dataloader import get_wsi_paths
import argparse
from tqdm import tqdm
import wandb
import os


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)

    # training parameters
    parser.add_argument('--batch_size', default=256, type=int,
                        help='batch size')
    parser.add_argument('--epochs', default=400, type=int,
                        help='number of epochs to pre-train')

    # patch extraction parameters
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--target_mpp', default=0.5, type=int,
                        help='resolution of the tiles')

    # encoder model parameters
    parser.add_argument('--patch_size', default=16, type=int,
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
    parser.add_argument('--masking_ratio', default=0.15, type=float,
                            help='masking ratio (percentage of removed patches)')
    parser.add_argument('--decoder_dim', default=512, type=int,
                        help='dim of the decoder')
    parser.add_argument('--decoder_depth', default=6, type=int,
                        help='depth of the decoder')

    # optimizer (AdamW) parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=1.5e-4, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--betas', type=float, default=(0.9, 0.95),
                        help='betas in AdamW')

    # scheduler parameters: decay with half-cosine after a warmup
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # dataset parameters TODO (not used yet)
    parser.add_argument('--data_paths', default='split.yml', type=str,
                        help='dataset path')

    # device parameters
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')

    # experiment parameters
    parser.add_argument('--run_name', default='test', help='name of the experiment run')
    parser.add_argument('--wandb_key', default='test', help='wand key used for logging')
    parser.add_argument('--experiment_dir', default='test', help='dir where to store results of this experiment')

    return parser


def pretrain(args):
    """
    Pre-train a MAE model for Barrett's tissue.
    """
    # set device
    device = torch.device(args.device)

    # get paths to training files
    tiff_files = get_wsi_paths(args.data_paths)

    # create a datasets of Barrett's tissue patches
    dataset = BarrettsTissueMAE(tiff_files=tiff_files, tile_size=(args.input_size, args.input_size), target_mpp=args.target_mpp)
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

    # log everything
    os.environ["WANDB_API_KEY"] = args.wandb_key
    wandb.init(
        project='Barret MAE pretraining',
        config={
            "image_size": args.input_size,
            "patch_size": args.patch_size,
            "num_classes": args.num_classes,
            "dim": args.encoder_dim,
            "depth": args.encoder_depth,
            "heads": args.encoder_heads,
            "mlp_dim": args.encoder_mlp_dim,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "epochs": args.epochs,
            "weight_decay": args.weight_decay})

    wandb.run.name = args.run_name

    # pre-train
    print('Starting pretraining for {} epochs.'.format(args.epochs))
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for image_batch in train_dataloader:
            loss = mae(image_batch.to(device))
            loss.backward()
            optimizer.step()
            epoch_loss += loss

        # per epoch lr scheduler
        lr_sched.adjust_learning_rate(optimizer, epoch, args)
        current_lr = optimizer.param_groups[0]["lr"]

        # print and log
        print('Epoch {}, loss: {:.3f}, lr: {}'.format(epoch,
                                                      epoch_loss / len(train_dataloader),
                                                      current_lr))
        wandb.log({'train loss': epoch_loss, 'lr': current_lr})

        # save the pre-trained encoder every 25 epochs
        if epoch % 25 == 0:
            torch.save(v.state_dict(), './trained-encoder/{}-epochs.pt'.format(epoch))

    wandb.finish()


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    pretrain(args)

