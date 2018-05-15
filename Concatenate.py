import package
import os
import numpy as np

# dataset = []
labels = []
one_hot = package.one_hot_encoder(51)
folders = os.listdir('./after_vgg19')

cls = 0
for folder in folders:
    files = os.listdir('./after_vgg19' + '/' + folder)
    for file in files:
        # features = np.load('./after_vgg19' + '/' + folder + '/' + file)
        # features = package.maximum_length_padding(features)  # leads to memory error
        # dataset.append(features)
        labels.append(np.array(one_hot[cls]))
    cls += 1

# dataset = np.array(dataset)
labels = np.array(labels)

print(labels.shape)

# np.save('./dataset_padded.npy', dataset)
np.save('./labels.npy', labels)
