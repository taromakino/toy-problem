import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
from data import N_CLASSES, N_ENVS
from torch.optim import Adam
from torchmetrics import Accuracy
from utils.enums import Task
from utils.nn_utils import MLP, arr_to_cov


IMAGE_EMBED_SHAPE = (32, 3, 3)
IMAGE_EMBED_SIZE = np.prod(IMAGE_EMBED_SHAPE)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.module_list = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.module_list(x)


class DCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.module_list = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
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
        self.mu_causal = MLP(IMAGE_EMBED_SIZE, h_sizes, N_ENVS * z_size)
        self.low_rank_causal = MLP(IMAGE_EMBED_SIZE, h_sizes, N_ENVS * z_size * rank)
        self.diag_causal = MLP(IMAGE_EMBED_SIZE, h_sizes, N_ENVS * z_size)
        self.mu_spurious = MLP(IMAGE_EMBED_SIZE, h_sizes, N_CLASSES * N_ENVS * z_size)
        self.low_rank_spurious = MLP(IMAGE_EMBED_SIZE, h_sizes, N_CLASSES * N_ENVS * z_size * rank)
        self.diag_spurious = MLP(IMAGE_EMBED_SIZE, h_sizes, N_CLASSES * N_ENVS * z_size)
        
    def causal_dist(self, x, e):
        batch_size = len(x)
        mu = self.mu_causal(x)
        mu = mu.reshape(batch_size, N_ENVS, self.z_size)
        mu = mu[torch.arange(batch_size), e, :]
        low_rank = self.low_rank_causal(x)
        low_rank = low_rank.reshape(batch_size, N_ENVS, self.z_size, self.rank)
        low_rank = low_rank[torch.arange(batch_size), e, :]
        diag = self.diag_causal(x)
        diag = diag.reshape(batch_size, N_ENVS, self.z_size)
        diag = diag[torch.arange(batch_size), e, :]
        cov = arr_to_cov(low_rank, diag)
        return D.MultivariateNormal(mu, cov)
    
    def spurious_params(self, x, y, e):
        batch_size = len(x)
        mu = self.mu_spurious(x)
        mu = mu.reshape(batch_size, N_CLASSES, N_ENVS, self.z_size)
        mu = mu[torch.arange(batch_size), y, e, :]
        low_rank = self.low_rank_spurious(x)
        low_rank = low_rank.reshape(batch_size, N_CLASSES, N_ENVS, self.z_size, self.rank)
        low_rank = low_rank[torch.arange(batch_size), y, e, :]
        diag = self.diag_spurious(x)
        diag = diag.reshape(batch_size, N_CLASSES, N_ENVS, self.z_size)
        diag = diag[torch.arange(batch_size), y, e, :]
        cov = arr_to_cov(low_rank, diag)
        return D.MultivariateNormal(mu, cov)

    def forward(self, x, y, e):
        batch_size = len(x)
        x = self.cnn(x).view(batch_size, -1)
        causal_dist = self.causal_dist(x, e)
        spurious_dist = self.spurious_params(x, y, e)
        return causal_dist, spurious_dist


class Decoder(nn.Module):
    def __init__(self, z_size, h_sizes):
        super().__init__()
        self.mlp = MLP(2 * z_size, h_sizes, IMAGE_EMBED_SIZE)
        self.dcnn = DCNN()

    def forward(self, x, z):
        batch_size = len(x)
        x_pred = self.mlp(z).reshape(batch_size, *IMAGE_EMBED_SHAPE)
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

    def causal_dist(self, e):
        mu = self.mu_causal[e]
        cov = arr_to_cov(self.low_rank_causal[e], self.diag_causal[e])
        return D.MultivariateNormal(mu, cov)

    def spurious_dist(self, y, e):
        mu = self.mu_spurious[y, e]
        cov = arr_to_cov(self.low_rank_spurious[y, e], self.diag_spurious[y, e])
        return D.MultivariateNormal(mu, cov)

    def forward(self, y, e):
        causal_dist = self.causal_dist(e)
        spurious_dist = self.spurious_dist(y, e)
        return causal_dist, spurious_dist


class VAE(pl.LightningModule):
    def __init__(self, task, z_size, rank, h_sizes, y_mult, beta, init_sd, kl_lb, lr, weight_decay, causal_mult,
            spurious_mult, lr_infer, n_infer_steps):
        super().__init__()
        self.save_hyperparameters()
        self.task = task
        self.z_size = z_size
        self.y_mult = y_mult
        self.beta = beta
        self.kl_lb = kl_lb
        self.lr = lr
        self.weight_decay = weight_decay
        self.causal_mult = causal_mult
        self.spurious_mult = spurious_mult
        self.lr_infer = lr_infer
        self.n_infer_steps = n_infer_steps
        # q(z|x,y,e)
        self.encoder = Encoder(z_size, rank, h_sizes)
        # p(x|z_c,z_s)
        self.decoder = Decoder(z_size, h_sizes)
        # p(z_c,z_s|y,e)
        self.prior = Prior(z_size, rank, init_sd)
        # p(y|z_c)
        self.classifier = MLP(z_size, h_sizes, 1)
        self.eval_metric = Accuracy('binary')

    def sample_z(self, causal_dist, spurious_dist):
        batch_size = len(causal_dist.loc)
        epsilon = torch.randn(batch_size, self.z_size, 1).to(self.device)
        mu_causal, tril_causal = causal_dist.loc, causal_dist.scale_tril
        mu_spurious, tril_spurious = spurious_dist.loc, spurious_dist.scale_tril
        z_c = mu_causal + torch.bmm(tril_causal, epsilon).squeeze()
        z_s = mu_spurious + torch.bmm(tril_spurious, epsilon).squeeze()
        return z_c, z_s

    def elbo(self, x, y, e):
        # z_c,z_s ~ q(z_c,z_s|x)
        posterior_causal, posterior_spurious = self.encoder(x, y, e)
        z_c, z_s = self.sample_z(posterior_causal, posterior_spurious)
        # E_q(z_c,z_s|x)[log p(x|z_c,z_s)]
        z = torch.hstack((z_c, z_s))
        log_prob_x_z = self.decoder(x, z).mean()
        # E_q(z_c|x)[log p(y|z_c)]
        y_pred = self.classifier(z_c).view(-1)
        log_prob_y_zc = -F.binary_cross_entropy_with_logits(y_pred, y.float())
        # KL(q(z_c,z_s|x) || p(z_c|e)p(z_s|y,e))
        prior_causal, prior_spurious = self.prior(y, e)
        kl_causal = D.kl_divergence(posterior_causal, prior_causal).mean()
        kl_spurious = D.kl_divergence(posterior_spurious, prior_spurious).mean()
        return log_prob_x_z, log_prob_y_zc, kl_causal, kl_spurious

    def training_step(self, batch, batch_idx):
        assert self.task == Task.VAE
        x, y, e, c, s = batch
        log_prob_x_z, log_prob_y_zc, kl_causal, kl_spurious = self.elbo(x, y, e)
        loss = -log_prob_x_z - self.y_mult * log_prob_y_zc + self.beta * (max(self.kl_lb, kl_causal) +
            max(self.kl_lb, kl_spurious))
        return loss

    def validation_step(self, batch, batch_idx):
        assert self.task == Task.VAE
        x, y, e, c, s = batch
        log_prob_x_z, log_prob_y_zc, kl_causal, kl_spurious = self.elbo(x, y, e)
        loss = -log_prob_x_z - self.y_mult * log_prob_y_zc + self.beta * (max(self.kl_lb, kl_causal) +
            max(self.kl_lb, kl_spurious))
        self.log('val_log_prob_x_z', log_prob_x_z, on_step=False, on_epoch=True)
        self.log('val_log_prob_y_zc', log_prob_y_zc, on_step=False, on_epoch=True)
        self.log('val_kl_causal', kl_causal, on_step=False, on_epoch=True)
        self.log('val_kl_spurious', kl_spurious, on_step=False, on_epoch=True)
        self.log('val_loss', loss, on_step=False, on_epoch=True)

    def infer_loss(self, x, y, e, z_c, z_s):
        # log p(x|z_c,z_s)
        z = torch.hstack((z_c, z_s))
        log_prob_x_z = self.decoder(x, z)
        # log p(y|z_c)
        y_pred = self.classifier(z_c).view(-1)
        log_prob_y_zc = -F.binary_cross_entropy_with_logits(y_pred, y.float(), reduction='none')
        # log p(z_c)
        posterior_causal, posterior_spurious = self.encoder(x, y, e)
        log_prob_zc = posterior_causal.log_prob(z_c)
        # log p(z_s)
        log_prob_zs = posterior_spurious.log_prob(z_s)
        loss = -log_prob_x_z - self.y_mult * log_prob_y_zc - self.causal_mult * log_prob_zc - self.spurious_mult * log_prob_zs
        return loss

    def classify(self, x):
        batch_size = len(x)
        loss_values = []
        y_values = []
        for y_value in range(N_CLASSES):
            y = torch.full((batch_size,), y_value, dtype=torch.long, device=self.device)
            for e_value in range(N_ENVS):
                e = torch.full((batch_size,), e_value, dtype=torch.long, device=self.device)
                posterior_causal, posterior_spurious = self.encoder(x, y, e)
                zc_param = nn.Parameter(posterior_causal.loc.detach())
                zs_param = nn.Parameter(posterior_spurious.loc.detach())
                y = torch.full((batch_size,), y_value, dtype=torch.long, device=self.device)
                optim = Adam([zc_param, zs_param], lr=self.lr_infer)
                for _ in range(self.n_infer_steps):
                    optim.zero_grad()
                    loss = self.infer_loss(x, y, e, zc_param, zs_param)
                    loss.mean().backward()
                    optim.step()
                loss_values.append(loss.detach().unsqueeze(-1))
                y_values.append(y_value)
        loss_values = torch.hstack(loss_values)
        y_values = torch.tensor(y_values, device=self.device)
        opt_loss = loss_values.min(dim=1)
        y_pred = y_values[opt_loss.indices]
        return opt_loss.values.mean(), y_pred

    def test_step(self, batch, batch_idx):
        assert self.task == Task.CLASSIFY
        x, y, e, c, s = batch
        with torch.set_grad_enabled(True):
            loss, y_pred = self.classify(x)
            self.log('loss', loss, on_step=False, on_epoch=True)
            self.eval_metric.update(y_pred, y)

    def on_test_epoch_end(self):
        assert self.task == Task.CLASSIFY
        self.log('eval_metric', self.eval_metric.compute())

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)