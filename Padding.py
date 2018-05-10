import numpy as np


def maximum_length_padding(original: np.ndarray):
    maximum_length = 1062
    patch = np.zeros([maximum_length - original.shape[0], 1000])
    print(original.shape)
    print(patch.shape)
    return np.concatenate((original, patch))
