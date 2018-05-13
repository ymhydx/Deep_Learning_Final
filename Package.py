import numpy as np


def maximum_length_padding(original: np.ndarray):
    maximum_length = 1062
    patch = np.zeros([maximum_length - original.shape[0], 1000])
    print(original.shape)
    print(patch.shape)
    return np.concatenate((original, patch))


def one_hot_encoder(num_classes: int):
    one_hot = {}
    for i in range(num_classes):
        code = np.zeros([1, num_classes])
        code[0, num_classes - 1 - i] = 1
        one_hot[i] = code
    return one_hot


def get(x: np.ndarray, y: np.ndarray, batch_szie):
    # x: dataset, dimension [num_samples,1062,1000]
    # y: labels, dimension [num_samples,1]
    num_samples = x.shape[0]
    random_seq = np.random.permutation(np.array(range(num_samples)))
    idx = random_seq[batch_szie]
    return x[idx], y[idx]