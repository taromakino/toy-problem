from enum import Enum


class Task(Enum):
    ALL = 'all'
    ERM_X = 'erm_x'
    ERM_C = 'erm_c'
    ERM_S = 'erm_s'
    VAE = 'vae'
    CLASSIFY = 'classify'


class EvalStage(Enum):
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'