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
from utils.nn_utils import MLP, one_hot, arr_to_cov, arr_to_tril


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
            nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1)
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
        self.mu = MLP(IMAGE_EMBED_SIZE + N_CLASSES + N_ENVS, h_sizes, 2 * z_size)
        self.low_rank = MLP(IMAGE_EMBED_SIZE + N_CLASSES + N_ENVS, h_sizes, 2 * z_size * 2 * rank)
        self.diag = MLP(IMAGE_EMBED_SIZE + N_CLASSES + N_ENVS, h_sizes, 2 * z_size)

    def forward(self, x, y, e):
        batch_size = len(x)
        x = self.cnn(x).view(batch_size, -1)
        y_one_hot = one_hot(y, N_CLASSES)
        e_one_hot = one_hot(e, N_ENVS)
        mu = self.mu(x, y_one_hot, e_one_hot)
        low_rank = self.low_rank(x, y_one_hot, e_one_hot)
        low_rank = low_rank.reshape(batch_size, 2 * self.z_size, 2 * self.rank)
        diag = self.diag(x, y_one_hot, e_one_hot)
        return D.MultivariateNormal(mu, scale_tril=arr_to_tril(low_rank, diag))


class Decoder(nn.Module):
    def __init__(self, z_size, h_sizes):
        super().__init__()
        self.mlp = MLP(2 * z_size, h_sizes, IMAGE_EMBED_SIZE)
        self.dcnn = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 2, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, x, z):
        batch_size = len(x)
        x_pred = self.mlp(z).reshape(batch_size, *IMAGE_EMBED_SHAPE)
        x_pred = self.dcnn(x_pred).view(batch_size, -1)
        return -F.binary_cross_entropy_with_logits(x_pred, x.view(batch_size, -1), reduction='none').sum(dim=1)


class Prior(nn.Module):
    def __init__(self, z_size, rank, h_sizes):
        super().__init__()
        self.z_size = z_size
        self.rank = rank
        self.mu_causal = MLP(N_ENVS, h_sizes, z_size)
        self.low_rank_causal = MLP(N_ENVS, h_sizes, z_size * rank)
        self.diag_causal = MLP(N_ENVS, h_sizes, z_size)
        # p(z_s|y,e)
        self.mu_spurious = MLP(N_CLASSES + N_ENVS, h_sizes, z_size)
        self.low_rank_spurious = MLP(N_CLASSES + N_ENVS, h_sizes, z_size * rank)
        self.diag_spurious = MLP(N_CLASSES + N_ENVS, h_sizes, z_size)

    def causal_params(self, e):
        batch_size = len(e)
        e_one_hot = one_hot(e, N_ENVS)
        mu = self.mu_causal(e_one_hot)
        low_rank = self.low_rank_causal(e_one_hot)
        low_rank = low_rank.reshape(batch_size, self.z_size, self.rank)
        diag = self.diag_causal(e_one_hot)
        cov = arr_to_cov(low_rank, diag)
        return mu, cov

    def spurious_params(self, y, e):
        batch_size = len(y)
        y_one_hot = one_hot(y, N_CLASSES)
        e_one_hot = one_hot(e, N_ENVS)
        mu = self.mu_spurious(y_one_hot, e_one_hot)
        low_rank = self.low_rank_spurious(y_one_hot, e_one_hot)
        low_rank = low_rank.reshape(batch_size, self.z_size, self.rank)
        diag = self.diag_spurious(y_one_hot, e_one_hot)
        cov = arr_to_cov(low_rank, diag)
        return mu, cov

    def log_prob_causal(self, z_c):
        batch_size = len(z_c)
        values = []
        for e_value in range(N_ENVS):
            e = torch.full((batch_size,), e_value, dtype=torch.long, device=z_c.device)
            dist = D.MultivariateNormal(*self.causal_params(e))
            values.append(dist.log_prob(z_c).unsqueeze(-1))
        values = torch.hstack(values)
        return torch.logsumexp(values, dim=1)

    def log_prob_spurious(self, z_s):
        batch_size = len(z_s)
        values = []
        for y_value in range(N_CLASSES):
            y = torch.full((batch_size,), y_value, dtype=torch.long, device=z_s.device)
            for e_value in range(N_ENVS):
                e = torch.full((batch_size,), e_value, dtype=torch.long, device=z_s.device)
                dist = D.MultivariateNormal(*self.spurious_params(y, e))
                values.append(dist.log_prob(z_s).unsqueeze(-1))
        values = torch.hstack(values)
        return torch.logsumexp(values, dim=1)

    def forward(self, y, e):
        batch_size = len(y)
        mu_causal, cov_causal = self.causal_params(e)
        mu_spurious, cov_spurious = self.spurious_params(y, e)
        mu = torch.hstack((mu_causal, mu_spurious))
        cov = torch.zeros(batch_size, 2 * self.z_size, 2 * self.z_size, device=y.device)
        cov[:, :self.z_size, :self.z_size] = cov_causal
        cov[:, self.z_size:, self.z_size:] = cov_spurious
        return D.MultivariateNormal(mu, cov)


class VAE(pl.LightningModule):
    def __init__(self, task, z_size, rank, h_sizes, y_mult, beta, reg_mult, lr, weight_decay, alpha, lr_infer,
            n_infer_steps):
        super().__init__()
        self.save_hyperparameters()
        self.task = task
        self.z_size = z_size
        self.y_mult = y_mult
        self.beta = beta
        self.reg_mult = reg_mult
        self.lr = lr
        self.weight_decay = weight_decay
        self.alpha = alpha
        self.lr_infer = lr_infer
        self.n_infer_steps = n_infer_steps
        # q(z_c|x,y,e)
        self.encoder = Encoder(z_size, rank, h_sizes)
        # p(x|z_c,z_s)
        self.decoder = Decoder(z_size, h_sizes)
        # p(z_c,z_s|y,e)
        self.prior = Prior(z_size, rank, h_sizes)
        # p(y|z_c)
        self.classifier = MLP(z_size, h_sizes, 1)
        self.eval_metric = Accuracy('binary')

    def sample_z(self, dist):
        mu, scale_tril = dist.loc, dist.scale_tril
        batch_size, z_size = mu.shape
        epsilon = torch.randn(batch_size, z_size, 1).to(self.device)
        return mu + torch.bmm(scale_tril, epsilon).squeeze()

    def loss(self, x, y, e):
        # z_c,z_s ~ q(z_c,z_s|x,y,e)
        posterior_dist = self.encoder(x, y, e)
        z = self.sample_z(posterior_dist)
        # E_q(z_c,z_s|x,y,e)[log p(x|z_c,z_s)]
        log_prob_x_z = self.decoder(x, z).mean()
        # E_q(z_c|x)[log p(y|z_c)]
        z_c, z_s = torch.chunk(z, 2, dim=1)
        y_pred = self.classifier(z_c).view(-1)
        log_prob_y_zc = -F.binary_cross_entropy_with_logits(y_pred, y.float())
        # KL(q(z_c,z_s|x) || p(z_c|e)p(z_s|y,e))
        prior_dist = self.prior(y, e)
        kl = D.kl_divergence(posterior_dist, prior_dist).mean()
        prior_reg = self.prior_reg(prior_dist)
        return log_prob_x_z, log_prob_y_zc, kl, prior_reg

    def prior_reg(self, prior_dist):
        mu = torch.zeros((1, 2 * self.z_size), device=self.device)
        cov = torch.eye(2 * self.z_size, device=self.device).unsqueeze(0)
        standard_normal = D.MultivariateNormal(mu, cov)
        return D.kl_divergence(prior_dist, standard_normal).mean()

    def training_step(self, batch, batch_idx):
        assert self.task == Task.VAE
        x, y, e, c, s = batch
        log_prob_x_z, log_prob_y_zc, kl, prior_reg = self.loss(x, y, e)
        loss = -log_prob_x_z - self.y_mult * log_prob_y_zc + self.beta * kl + self.reg_mult * prior_reg
        return loss

    def validation_step(self, batch, batch_idx):
        assert self.task == Task.VAE
        x, y, e, c, s = batch
        log_prob_x_z, log_prob_y_zc, kl, prior_reg = self.loss(x, y, e)
        loss = -log_prob_x_z - self.y_mult * log_prob_y_zc + self.beta * kl + self.reg_mult * prior_reg
        self.log('val_log_prob_x_z', log_prob_x_z, on_step=False, on_epoch=True)
        self.log('val_log_prob_y_zc', log_prob_y_zc, on_step=False, on_epoch=True)
        self.log('val_kl', kl, on_step=False, on_epoch=True)
        self.log('val_loss', loss, on_step=False, on_epoch=True)

    def infer_loss(self, x, y, z):
        # log p(x|z_c,z_s)
        log_prob_x_z = self.decoder(x, z)
        # log p(y|z_c)
        z_c, z_s = torch.chunk(z, 2, dim=1)
        y_pred = self.classifier(z_c).view(-1)
        log_prob_y_zc = -F.binary_cross_entropy_with_logits(y_pred, y.float(), reduction='none')
        # log p(z)
        log_prob_zc = self.prior.log_prob_causal(z_c)
        log_prob_zs = self.prior.log_prob_spurious(z_s)
        log_prob_z = log_prob_zc + log_prob_zs
        loss = -log_prob_x_z - self.y_mult * log_prob_y_zc - self.alpha * log_prob_z
        return loss

    def make_z_param(self, batch_size, y_value, e_value):
        y = torch.full((batch_size,), y_value, dtype=torch.long, device=self.device)
        e = torch.full((batch_size,), e_value, dtype=torch.long, device=self.device)
        causal_dist = D.MultivariateNormal(*self.prior.causal_params(e))
        spurious_dist = D.MultivariateNormal(*self.prior.spurious_params(y, e))
        return nn.Parameter(torch.hstack((causal_dist.loc, spurious_dist.loc)))

    def opt_infer_loss(self, x, y_value):
        batch_size = len(x)
        z_param = nn.Parameter(torch.zeros(batch_size, 2 * self.z_size, device=self.device))
        nn.init.normal_(z_param)
        y = torch.full((batch_size,), y_value, dtype=torch.long, device=self.device)
        optim = Adam([z_param], lr=self.lr_infer)
        for _ in range(self.n_infer_steps):
            optim.zero_grad()
            loss = self.infer_loss(x, y, z_param)
            loss.mean().backward()
            optim.step()
        return loss.detach().clone()

    def infer_z(self, x):
        loss_candidates = []
        y_candidates = []
        for y_value in range(N_CLASSES):
            loss_candidates.append(self.opt_infer_loss(x, y_value)[:, None])
            y_candidates.append(y_value)
        loss_candidates = torch.hstack(loss_candidates)
        y_candidates = torch.tensor(y_candidates, device=self.device)
        opt_loss = loss_candidates.min(dim=1)
        y_pred = y_candidates[opt_loss.indices]
        return opt_loss.values.mean(), y_pred

    def test_step(self, batch, batch_idx):
        assert self.task == Task.CLASSIFY
        x, y, e, c, s = batch
        with torch.set_grad_enabled(True):
            loss, y_pred = self.infer_z(x)
            self.log('loss', loss, on_step=False, on_epoch=True)
            self.eval_metric.update(y_pred, y)

    def on_test_epoch_end(self):
        assert self.task == Task.CLASSIFY
        self.log('eval_metric', self.eval_metric.compute())

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)