import numpy as np


# pad each video to the same length (maximum length)
def subsample(original: np.ndarray, aim: int):
    length = original.shape[0]
    step = int(length / aim)
    indexs = [step * x for x in range(aim)]
    return original[indexs]


# one-hot encoding
def one_hot_encoder(num_classes: int):
    one_hot = {}
    for i in range(num_classes):
        code = np.zeros(num_classes)
        code[num_classes - 1 - i] = 1
        one_hot[i] = code
    return one_hot


# given batch size, return data and labels
def get(x: np.ndarray, y: np.ndarray, batch_szie: int):
    # x: dataset, dimension [num_samples,1062,1000]
    # y: labels, dimension [num_samples,1]
    num_samples = x.shape[0]
    random_seq = np.random.permutation(np.array(range(num_samples)))
    idx = random_seq[:batch_szie]
    return x[idx], y[idx]


# generate training set, validation set and test set
def split(set: np.ndarray):
    # randomize
    set = set[np.random.permutation(np.array(range(set.shape[0])))]
    # train/validation/test=80%/10%/10%
    train_num = int(set.shape[0] * 0.8)
    validation_num = int((set.shape[0] - train_num) / 2)
    train_set, validation_set, test_set = set[:train_num], set[train_num:validation_num], set[
        validation_num:]
    return train_set, validation_set, test_set

# encode probability with one-hot
def prob2one_hot(prob: np.ndarray):
    for i in range(prob.shape[0]):
        max_value = np.amax(prob[i, :])
        for j in range(prob.shape[1]):
            if abs(prob[i, j] - max_value) < 1e-9:
                prob[i, j] = 1
            else:
                prob[i, j] = 0

    return prob

# calculate accuracy
def accuracy(prediction: np.ndarray, truth: np.ndarray):
    num = prediction.shape[0]
    num_correct = 0
    for i in range(num):
        if np.array_equal(prediction[i, :], truth[i, :]):
            num_correct += 1

    return num_correct / num
