import matplotlib.pyplot as plt
import numpy as np
import os
import pytorch_lightning as pl
import torch
from data import N_CLASSES, N_ENVS, MAKE_DATA, PLOT
from models.vae import IMAGE_EMBED_SHAPE, VAE
from utils.enums import Task


N_EXAMPLES = 10
N_COLS = 10


def sample_prior(rng, model):
    y = torch.tensor(rng.choice(N_CLASSES), dtype=torch.long, device=model.device)[None]
    e = torch.tensor(rng.choice(N_ENVS), dtype=torch.long, device=model.device)[None]
    prior_dist = model.prior(y, e)
    z_sample = prior_dist.sample()
    return torch.chunk(z_sample, 2, dim=1)


def reconstruct_x(model, z):
    batch_size = len(z)
    x_pred = model.decoder.mlp(z).reshape(batch_size, *IMAGE_EMBED_SHAPE)
    x_pred = model.decoder.dcnn(x_pred)
    return torch.sigmoid(x_pred)


def main(args):
    rng = np.random.RandomState(args.seed)
    task_dpath = os.path.join(args.dpath, Task.VAE.value)
    pl.seed_everything(args.seed)
    dataloader, _, _ = MAKE_DATA[args.dataset](args.train_ratio, args.batch_size)
    model = VAE.load_from_checkpoint(os.path.join(task_dpath, f'version_{args.seed}', 'checkpoints', 'best.ckpt'))
    x, y, e, c, s = dataloader.dataset[:]
    for example_idx in range(N_EXAMPLES):
        x_seed = x[[example_idx]].to(model.device)
        posterior_dist_seed = model.encoder(x_seed)
        z_seed = posterior_dist_seed.loc
        zc_seed, zs_seed = torch.chunk(z_seed, 2, dim=1)
        fig, axes = plt.subplots(2, N_COLS, figsize=(2 * N_COLS, 2 * 2))
        for ax in axes.flatten():
            ax.set_xticks([])
            ax.set_yticks([])
        plot = PLOT[args.dataset]
        plot(axes[0, 0], x_seed.squeeze().cpu().numpy())
        plot(axes[1, 0], x_seed.squeeze().cpu().numpy())
        x_pred = reconstruct_x(model, z_seed)
        plot(axes[0, 1], x_pred.squeeze().detach().cpu().numpy())
        plot(axes[1, 1], x_pred.squeeze().detach().cpu().numpy())
        for col_idx in range(2, N_COLS):
            zc_sample, zs_sample = sample_prior(rng, model)
            x_pred_causal = reconstruct_x(model, torch.hstack((zc_sample, zs_seed)))
            x_pred_spurious = reconstruct_x(model, torch.hstack((zc_seed, zs_sample)))
            plot(axes[0, col_idx], x_pred_causal.squeeze().detach().cpu().numpy())
            plot(axes[1, col_idx], x_pred_spurious.squeeze().detach().cpu().numpy())
        fig_dpath = os.path.join(task_dpath, f'version_{args.seed}', 'fig', 'plot_reconstruction')
        os.makedirs(fig_dpath, exist_ok=True)
        plt.savefig(os.path.join(fig_dpath, f'{example_idx}.png'))
        plt.close()