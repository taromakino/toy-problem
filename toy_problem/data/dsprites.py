import matplotlib.pyplot as plt
import numpy as np
import torch
from utils.nn_utils import make_dataloader
from utils.plot import hist_discrete


RNG = np.random.RandomState(0)
PROB_ZERO_E0 = 0.25
N_TRAINVAL = 10000
N_TEST = 2000
WIDTH_SMALL = 4
WIDTH_LARGE = 12
IMAGE_SIZE = 28


def flip_binary(x, flip_prob):
    idxs = np.arange(len(x))
    flip_idxs = RNG.choice(idxs, size=int(flip_prob * len(idxs)), replace=False)
    x[flip_idxs] = 1 - x[flip_idxs]
    return x


def make_trainval_data():
    scale = RNG.randint(0, 2, N_TRAINVAL)

    y = flip_binary(scale.copy(), 0.25)

    idxs_env0 = []
    zero_idxs = np.where(scale == 0)[0]
    one_idxs = np.where(scale == 1)[0]
    idxs_env0.append(RNG.choice(zero_idxs, size=int(PROB_ZERO_E0 * len(zero_idxs))))
    idxs_env0.append(RNG.choice(one_idxs, size=int((1 - PROB_ZERO_E0) * len(one_idxs))))
    idxs_env0 = np.concatenate(idxs_env0)
    idxs_env1 = np.setdiff1d(np.arange(N_TRAINVAL), idxs_env0)

    e = np.zeros(N_TRAINVAL, dtype='long')
    e[idxs_env1] = 1

    colors = np.full(N_TRAINVAL, np.nan)
    idxs_y0_e0 = np.where((y == 0) & (e == 0))[0]
    idxs_y1_e0 = np.where((y == 1) & (e == 0))[0]
    idxs_y0_e1 = np.where((y == 0) & (e == 1))[0]
    idxs_y1_e1 = np.where((y == 1) & (e == 1))[0]
    colors[idxs_y0_e0] = RNG.normal(0.2, 0.1, len(idxs_y0_e0))
    colors[idxs_y1_e0] = RNG.normal(0.6, 0.1, len(idxs_y1_e0))
    colors[idxs_y0_e1] = RNG.normal(0.8, 0.1, len(idxs_y0_e1))
    colors[idxs_y1_e1] = RNG.normal(0.4, 0.1, len(idxs_y1_e1))
    colors = np.clip(colors, 0, 1)[:, None, None]

    center_x = RNG.randint(WIDTH_LARGE // 2, IMAGE_SIZE - WIDTH_LARGE // 2 + 1, N_TRAINVAL)
    center_y = RNG.randint(WIDTH_LARGE // 2, IMAGE_SIZE - WIDTH_LARGE // 2 + 1, N_TRAINVAL)

    x = np.zeros((N_TRAINVAL, IMAGE_SIZE, IMAGE_SIZE))
    for idx in range(N_TRAINVAL):
        width = WIDTH_SMALL if scale[idx] == 0 else WIDTH_LARGE
        half_width_floor = np.floor(width / 2)
        half_width_ceil = np.ceil(width / 2)
        x_lb = int(center_x[idx] - half_width_floor)
        x_ub = int(center_x[idx] + half_width_ceil)
        y_lb = int(center_y[idx] - half_width_floor)
        y_ub = int(center_y[idx] + half_width_ceil)
        x[idx, x_lb:x_ub, y_lb:y_ub] = 1

    x = np.stack([x, x], axis=1)
    x[:, 0, :, :] *= colors
    x[:, 1, :, :] *= (1 - colors)
    x = torch.tensor(x, dtype=torch.float32)

    y = torch.tensor(y, dtype=torch.long)
    e = torch.tensor(e, dtype=torch.long)
    c = torch.tensor(scale, dtype=torch.float32)
    s = torch.tensor(colors.squeeze(), dtype=torch.float32)
    return x, y, e, c, s


def make_test_data():
    scale = RNG.randint(0, 2, N_TEST)

    y = flip_binary(scale.copy(), 0.25)

    colors = RNG.random(N_TEST)[:, None, None]

    center_x = RNG.randint(WIDTH_LARGE // 2, IMAGE_SIZE - WIDTH_LARGE // 2 + 1, N_TRAINVAL)
    center_y = RNG.randint(WIDTH_LARGE // 2, IMAGE_SIZE - WIDTH_LARGE // 2 + 1, N_TRAINVAL)

    x = np.zeros((N_TEST, IMAGE_SIZE, IMAGE_SIZE))
    for idx in range(N_TEST):
        width = WIDTH_SMALL if scale[idx] == 0 else WIDTH_LARGE
        half_width_floor = np.floor(width / 2)
        half_width_ceil = np.ceil(width / 2)
        x_lb = int(center_x[idx] - half_width_floor)
        x_ub = int(center_x[idx] + half_width_ceil)
        y_lb = int(center_y[idx] - half_width_floor)
        y_ub = int(center_y[idx] + half_width_ceil)
        x[idx, x_lb:x_ub, y_lb:y_ub] = 1

    x = np.stack([x, x], axis=1)
    x[:, 0, :, :] *= colors
    x[:, 1, :, :] *= (1 - colors)
    x = torch.tensor(x, dtype=torch.float32)

    y = torch.tensor(y, dtype=torch.long)
    e = torch.full_like(y, np.nan, dtype=torch.float32)
    c = torch.tensor(scale, dtype=torch.float32)
    s = torch.tensor(colors.squeeze(), dtype=torch.float32)
    return x, y, e, c, s


def subsample(x, y, e, c, s, n_examples):
    idxs = RNG.choice(len(x), n_examples, replace=False)
    return x[idxs], y[idxs], e[idxs], c[idxs], s[idxs]


def make_data(train_ratio, batch_size):
    x, y, e, c, s = make_trainval_data()
    n_total = len(e)
    n_train = int(train_ratio * n_total)
    train_idxs = RNG.choice(np.arange(n_total), n_train, replace=False)
    val_idxs = np.setdiff1d(np.arange(n_total), train_idxs)
    x_train, y_train, e_train, c_train, s_train = x[train_idxs], y[train_idxs], e[train_idxs], c[train_idxs], s[train_idxs]
    x_val, y_val, e_val, c_val, s_val = x[val_idxs], y[val_idxs], e[val_idxs], c[val_idxs], s[val_idxs]
    x_test, y_test, e_test, c_test, s_test = make_test_data()
    data_train = make_dataloader((x_train, y_train, e_train, c_train, s_train), batch_size, True)
    data_val = make_dataloader((x_val, y_val, e_val, c_val, s_val), batch_size, False)
    data_test = make_dataloader((x_test, y_test, e_test, c_test, s_test), batch_size, False)
    return data_train, data_val, data_test


def main():
    x, y, e, c, s = make_trainval_data()
    scale = c
    colors = s
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    hist_discrete(axes[0], scale[e == 0])
    hist_discrete(axes[1], scale[e == 1])
    axes[0].set_title('p(width|e=0)')
    axes[1].set_title('p(width|e=1)')
    fig.suptitle('Assumed Gaussian')
    fig.tight_layout()
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    axes[0].hist(colors[(y == 0) & (e == 0)], bins='auto')
    axes[1].hist(colors[(y == 0) & (e == 1)], bins='auto')
    axes[2].hist(colors[(y == 1) & (e == 0)], bins='auto')
    axes[3].hist(colors[(y == 1) & (e == 1)], bins='auto')
    axes[0].set_title('p(color|y=0,e=0)')
    axes[1].set_title('p(color|y=0,e=1)')
    axes[2].set_title('p(color|y=1,e=0)')
    axes[3].set_title('p(color|y=1,e=1)')
    fig.suptitle('Assumed Gaussian')
    fig.tight_layout()
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    axes[0].hist(scale[(y == 0) & (e == 0)])
    axes[1].hist(scale[(y == 0) & (e == 1)])
    axes[2].hist(scale[(y == 1) & (e == 0)])
    axes[3].hist(scale[(y == 1) & (e == 1)])
    axes[0].set_title('p(size|y=0,e=0)')
    axes[1].set_title('p(size|y=0,e=1)')
    axes[2].set_title('p(size|y=1,e=0)')
    axes[3].set_title('p(size|y=1,e=1)')
    fig.suptitle('Assumed Non-Gaussian')
    fig.tight_layout()
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].hist(colors[e == 0], bins='auto')
    axes[1].hist(colors[e == 1], bins='auto')
    axes[0].set_title('p(color|e=0)')
    axes[1].set_title('p(color|e=1)')
    fig.suptitle('Assumed Non-Gaussian')
    fig.tight_layout()
    plt.show(block=True)


if __name__ == '__main__':
    main()