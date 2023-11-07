import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


sns.set_context(context='paper', font_scale=2)


def next_color(ax):
    return next(ax._get_lines.prop_cycler)['color']


def hist_discrete(ax, x):
    n_bins = len(np.unique(x))
    ax.hist(x, bins=n_bins)


def plot_red_green_image(ax, image):
    '''
    Input image has shape (2, m, n)
    '''
    _, m, n = image.shape
    image = image.transpose((1, 2, 0))
    image = np.dstack((image, np.zeros((m, n))))
    ax.imshow(image)