import package
import os
import numpy as np

#kept frames
num_frames=70

dataset = []
labels = []
one_hot = package.one_hot_encoder(51)
folders = os.listdir('./vgg19_augmented')

cls = 0
for folder in folders:
    files = os.listdir('./vgg19_augmented' + '/' + folder)
    for file in files:
        key=True
        for sign in file.split('_'):
            if sign=='30' or sign=='40':
                key=False
        if key:
            features = np.load('./vgg19_augmented' + '/' + folder + '/' + file)
            features = package.subsample(features,num_frames)
            dataset.append(features)
            labels.append(np.array(one_hot[cls]))
    cls += 1

dataset = np.array(dataset)
labels = np.array(labels)

np.save('./features_augmented.npy', dataset)
np.save('./labels_augmented.npy', labels)
