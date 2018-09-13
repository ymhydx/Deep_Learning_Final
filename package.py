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
