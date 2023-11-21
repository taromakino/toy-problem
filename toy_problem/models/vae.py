import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
from data import N_CLASSES, N_ENVS
from torch.optim import Adam
from torchmetrics import Accuracy
from utils.nn_utils import MLP, arr_to_cov


IMG_EMBED_SHAPE = (32, 3, 3)
IMG_EMBED_SIZE = np.prod(IMG_EMBED_SHAPE)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.module_list = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32)
        )

    def forward(self, x):
        return self.module_list(x)


class DCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.module_list = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 2, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, x):
        return self.module_list(x)


class Encoder(nn.Module):
    def __init__(self, z_size, rank, h_sizes):
        super().__init__()
        self.z_size = z_size
        self.rank = rank
        self.cnn = CNN()
        self.mu_causal = MLP(IMG_EMBED_SIZE, h_sizes, N_ENVS * z_size)
        self.low_rank_causal = MLP(IMG_EMBED_SIZE, h_sizes, N_ENVS * z_size * rank)
        self.diag_causal = MLP(IMG_EMBED_SIZE, h_sizes, N_ENVS * z_size)
        self.mu_spurious = MLP(IMG_EMBED_SIZE, h_sizes, N_CLASSES * N_ENVS * z_size)
        self.low_rank_spurious = MLP(IMG_EMBED_SIZE, h_sizes, N_CLASSES * N_ENVS * z_size * rank)
        self.diag_spurious = MLP(IMG_EMBED_SIZE, h_sizes, N_CLASSES * N_ENVS * z_size)

    def forward(self, x, y, e):
        batch_size = len(x)
        x = self.cnn(x).view(batch_size, -1)
        # Causal
        mu_causal = self.mu_causal(x)
        mu_causal = mu_causal.reshape(batch_size, N_ENVS, self.z_size)
        mu_causal = mu_causal[torch.arange(batch_size), e, :]
        low_rank_causal = self.low_rank_causal(x)
        low_rank_causal = low_rank_causal.reshape(batch_size, N_ENVS, self.z_size, self.rank)
        low_rank_causal = low_rank_causal[torch.arange(batch_size), e, :]
        diag_causal = self.diag_causal(x)
        diag_causal = diag_causal.reshape(batch_size, N_ENVS, self.z_size)
        diag_causal = diag_causal[torch.arange(batch_size), e, :]
        cov_causal = arr_to_cov(low_rank_causal, diag_causal)
        # Spurious
        mu_spurious = self.mu_spurious(x)
        mu_spurious = mu_spurious.reshape(batch_size, N_CLASSES, N_ENVS, self.z_size)
        mu_spurious = mu_spurious[torch.arange(batch_size), y, e, :]
        low_rank_spurious = self.low_rank_spurious(x)
        low_rank_spurious = low_rank_spurious.reshape(batch_size, N_CLASSES, N_ENVS, self.z_size, self.rank)
        low_rank_spurious = low_rank_spurious[torch.arange(batch_size), y, e, :]
        diag_spurious = self.diag_spurious(x)
        diag_spurious = diag_spurious.reshape(batch_size, N_CLASSES, N_ENVS, self.z_size)
        diag_spurious = diag_spurious[torch.arange(batch_size), y, e, :]
        cov_spurious = arr_to_cov(low_rank_spurious, diag_spurious)
        # Block diagonal
        mu = torch.hstack((mu_causal, mu_spurious))
        cov = torch.zeros(batch_size, 2 * self.z_size, 2 * self.z_size, device=y.device)
        cov[:, :self.z_size, :self.z_size] = cov_causal
        cov[:, self.z_size:, self.z_size:] = cov_spurious
        return D.MultivariateNormal(mu, scale_tril=torch.linalg.cholesky(cov))


class Decoder(nn.Module):
    def __init__(self, z_size, h_sizes):
        super().__init__()
        self.mlp = MLP(2 * z_size, h_sizes, IMG_EMBED_SIZE)
        self.dcnn = DCNN()

    def forward(self, x, z):
        batch_size = len(x)
        x_pred = self.mlp(z).view(batch_size, *IMG_EMBED_SHAPE)
        x_pred = self.dcnn(x_pred).view(batch_size, -1)
        return -F.binary_cross_entropy_with_logits(x_pred, x.view(batch_size, -1), reduction='none').sum(dim=1)


class Prior(nn.Module):
    def __init__(self, z_size, rank, init_sd):
        super().__init__()
        self.z_size = z_size
        self.mu_causal = nn.Parameter(torch.zeros(N_ENVS, z_size))
        self.low_rank_causal = nn.Parameter(torch.zeros(N_ENVS, z_size, rank))
        self.diag_causal = nn.Parameter(torch.zeros(N_ENVS, z_size))
        nn.init.normal_(self.mu_causal, 0, init_sd)
        nn.init.normal_(self.low_rank_causal, 0, init_sd)
        nn.init.normal_(self.diag_causal, 0, init_sd)
        # p(z_s|y,e)
        self.mu_spurious = nn.Parameter(torch.zeros(N_CLASSES, N_ENVS, z_size))
        self.low_rank_spurious = nn.Parameter(torch.zeros(N_CLASSES, N_ENVS, z_size, rank))
        self.diag_spurious = nn.Parameter(torch.zeros(N_CLASSES, N_ENVS, z_size))
        nn.init.normal_(self.mu_spurious, 0, init_sd)
        nn.init.normal_(self.low_rank_spurious, 0, init_sd)
        nn.init.normal_(self.diag_spurious, 0, init_sd)

    def forward(self, y, e):
        batch_size = len(y)
        # Causal
        mu_causal = self.mu_causal[e]
        cov_causal = arr_to_cov(self.low_rank_causal[e], self.diag_causal[e])
        # Spurious
        mu_spurious = self.mu_spurious[y, e]
        cov_spurious = arr_to_cov(self.low_rank_spurious[y, e], self.diag_spurious[y, e])
        # Block diagonal
        mu = torch.hstack((mu_causal, mu_spurious))
        cov = torch.zeros(batch_size, 2 * self.z_size, 2 * self.z_size, device=y.device)
        cov[:, :self.z_size, :self.z_size] = cov_causal
        cov[:, self.z_size:, self.z_size:] = cov_spurious
        return D.MultivariateNormal(mu, cov)


class VAE(pl.LightningModule):
    def __init__(self, task, z_size, rank, h_sizes, y_mult, beta, reg_mult, init_sd, n_samples, lr, weight_decay,
            lr_infer, n_infer_steps):
        super().__init__()
        self.save_hyperparameters()
        self.task = task
        self.z_size = z_size
        self.y_mult = y_mult
        self.beta = beta
        self.reg_mult = reg_mult
        self.n_samples = n_samples
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_infer = lr_infer
        self.n_infer_steps = n_infer_steps
        # q(z_c,z_s|x)
        self.encoder = Encoder(z_size, rank, h_sizes)
        # p(x|z_c, z_s)
        self.decoder = Decoder(z_size, h_sizes)
        # p(z_c,z_s|y,e)
        self.prior = Prior(z_size, rank, init_sd)
        # p(y|z)
        self.classifier = MLP(z_size, h_sizes, 1)
        self.z_mu_samples = []
        self.z_cov_samples = []
        self.z_mu = nn.Parameter(torch.zeros(n_samples, 2 * z_size))
        self.z_cov = nn.Parameter(torch.eye(2 * z_size).unsqueeze(0).repeat_interleave(n_samples, dim=0))
        self.z_mu.requires_grad_(False)
        self.z_cov.requires_grad_(False)
        self.val_acc = Accuracy('binary')
        self.test_acc = Accuracy('binary')

    def sample_z(self, dist):
        mu, scale_tril = dist.loc, dist.scale_tril
        batch_size, z_size = mu.shape
        epsilon = torch.randn(batch_size, z_size, 1).to(self.device)
        return mu + torch.bmm(scale_tril, epsilon).squeeze()

    def loss(self, x, y, e):
        # z_c,z_s ~ q(z_c,z_s|x)
        posterior_dist = self.encoder(x, y, e)
        z = self.sample_z(posterior_dist)
        # E_q(z_c,z_s|x)[log p(x|z_c,z_s)]
        log_prob_x_z = self.decoder(x, z).mean()
        # E_q(z_c|x)[log p(y|z_c)]
        z_c, z_s = torch.chunk(z, 2, dim=1)
        y_pred = self.classifier(z_c).view(-1)
        log_prob_y_zc = -F.binary_cross_entropy_with_logits(y_pred, y.float())
        # KL(q(z_c,z_s|x) || p(z_c|e)p(z_s|y,e))
        prior_dist = self.prior(y, e)
        kl = D.kl_divergence(posterior_dist, prior_dist).mean()
        prior_norm = (prior_dist.loc ** 2).mean()
        self.z_mu_samples.append(posterior_dist.loc.detach().cpu())
        self.z_cov_samples.append(posterior_dist.covariance_matrix.detach().cpu())
        return log_prob_x_z, log_prob_y_zc, kl, prior_norm

    def training_step(self, batch, batch_idx):
        x, y, e, c, s = batch
        log_prob_x_z, log_prob_y_zc, kl, prior_norm = self.loss(x, y, e)
        loss = -log_prob_x_z - self.y_mult * log_prob_y_zc + self.beta * kl + self.reg_mult * prior_norm
        self.log('train_log_prob_x_z', log_prob_x_z, on_step=False, on_epoch=True)
        self.log('train_log_prob_y_zc', log_prob_y_zc, on_step=False, on_epoch=True)
        self.log('train_kl', kl, on_step=False, on_epoch=True)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def on_train_epoch_start(self):
        self.z_mu_samples = []
        self.z_cov_samples = []

    def on_train_epoch_end(self):
        self.z_mu_samples = torch.cat(self.z_mu_samples)
        self.z_cov_samples = torch.cat(self.z_cov_samples)
        rng = np.random.RandomState(self.trainer.logger.version)
        idxs = rng.choice(len(self.z_mu_samples), self.n_samples, replace=False)
        self.z_mu.data = self.z_mu_samples[idxs]
        self.z_cov.data = self.z_cov_samples[idxs]

    def infer_loss(self, x, y, z):
        # log p(x|z_c,z_s)
        log_prob_x_z = self.decoder(x, z)
        # log p(y|z_c)
        z_c, z_s = torch.chunk(z, 2, dim=1)
        y_pred = self.classifier(z_c).view(-1)
        log_prob_y_zc = -F.binary_cross_entropy_with_logits(y_pred, y.float(), reduction='none')
        # log q(z)
        log_prob_z = D.MultivariateNormal(self.z_mu.to(self.device), self.z_cov.to(self.device)).log_prob(z.unsqueeze(1))
        log_prob_z = torch.logsumexp(log_prob_z, dim=1)
        loss = -log_prob_x_z - self.y_mult * log_prob_y_zc - log_prob_z
        return loss

    def opt_infer_loss(self, x, y_value):
        batch_size = len(x)
        z_mu_rep = self.z_mu.mean(dim=0).unsqueeze(0).repeat_interleave(batch_size, dim=0)
        z_param = nn.Parameter(z_mu_rep.detach().to(self.device))
        y = torch.full((batch_size,), y_value, dtype=torch.long, device=self.device)
        optim = Adam([z_param], lr=self.lr_infer)
        for _ in range(self.n_infer_steps):
            optim.zero_grad()
            loss = self.infer_loss(x, y, z_param)
            loss.mean().backward()
            optim.step()
        return loss.detach().clone()

    def infer_z(self, x):
        loss_values = []
        for y_value in range(N_CLASSES):
            loss_values.append(self.opt_infer_loss(x, y_value)[:, None])
        loss_values = torch.hstack(loss_values)
        y_pred = loss_values.argmin(dim=1)
        return y_pred

    def validation_step(self, batch, batch_idx, dataloader_idx):
        x, y, e, c, s = batch
        with torch.set_grad_enabled(True):
            y_pred = self.infer_z(x)
            if dataloader_idx == 0:
                self.val_acc.update(y_pred, y)
            elif dataloader_idx == 1:
                self.test_acc.update(y_pred, y)
            else:
                raise ValueError

    def on_validation_epoch_end(self):
        self.log('val_acc', self.val_acc.compute())
        self.log('test_acc', self.test_acc.compute())

    def test_step(self, batch, batch_idx):
        x, y, e, c, s = batch
        with torch.set_grad_enabled(True):
            y_pred = self.infer_z(x)
            self.test_acc.update(y_pred, y)

    def on_test_epoch_end(self):
        self.log('test_acc', self.test_acc.compute())

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)