def min_max_scale(x):
    return (x - x.min()) / (x.max() - x.min())