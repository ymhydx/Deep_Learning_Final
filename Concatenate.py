import Package
import os
import numpy as np

dataset = []
labels = []
one_hot = Package.one_hot_encoder(51)
folders = os.listdir('./after_vgg19')

cls = 0
for folder in folders:
    files = os.listdir('./after_vgg19' + '/' + folder)
    for file in files:
        features = np.load('./after_vgg19' + '/' + folder + '/' + file)
        dataset.append(features)
        labels.append(np.array(one_hot[cls]))
    cls += 1

dataset = np.array(dataset)
labels = np.array(labels)

np.save('./dataset.npy', dataset)
np.save('./labels.npy', labels)
