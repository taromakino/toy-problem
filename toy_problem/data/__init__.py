import data.cmnist
import data.dsprites
from utils.plot import plot_red_green_image


N_CLASSES = 2
N_ENVS = 2


MAKE_DATA = {
    'cmnist': data.cmnist.make_data,
    'dsprites': data.dsprites.make_data
}


PLOT = {
    'cmnist': plot_red_green_image,
    'dsprites': plot_red_green_image
}