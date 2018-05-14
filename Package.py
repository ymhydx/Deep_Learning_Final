import numpy as np


# pad each video to the same length (maximum length)
def maximum_length_padding(original: np.ndarray):
    maximum_length = 1062
    patch = np.zeros([maximum_length - original.shape[0], 1000])
    print(original.shape)
    print(patch.shape)
    return np.concatenate((original, patch))


# one-hot encoding
def one_hot_encoder(num_classes: int):
    one_hot = {}
    for i in range(num_classes):
        code = np.zeros([1, num_classes])
        code[0, num_classes - 1 - i] = 1
        one_hot[i] = code
    return one_hot


# given batch size, return data and labels
def get(x: np.ndarray, y: np.ndarray, batch_szie: int):
    # x: dataset, dimension [num_samples,1062,1000]
    # y: labels, dimension [num_samples,1]
    num_samples = x.shape[0]
    random_seq = np.random.permutation(np.array(range(num_samples)))
    idx = random_seq[batch_szie]
    return x[idx], y[idx]


# generate training set, validation set and test set
def divide_set(set: np.ndarray):
    # randomize
    set = set[np.random.permutation(np.array(range(set.shape[0])))]
    # train/validation/test=50%/25%/25%
    train_num = int(set.shape[0] / 2)
    validation_num = int((set.shape[0] - train_num) / 2)
    train_set, validation_set, test_set = set[:train_num], set[train_num:validation_num], set[
                                                                                          validation_num:]
    return train_set, validation_set, test_set
