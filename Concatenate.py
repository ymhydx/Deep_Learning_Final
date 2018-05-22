import package
import os
import numpy as np

#kept frames
num_frames=300

dataset = []
labels = []
one_hot = package.one_hot_encoder(51)
folders = os.listdir('./after_vgg19')

cls = 0
for folder in folders:
    files = os.listdir('./after_vgg19' + '/' + folder)
    for file in files:
        features = np.load('./after_vgg19' + '/' + folder + '/' + file)
        features = package.subsample(features,num_frames)
        dataset.append(features)
        labels.append(np.array(one_hot[cls]))
    cls += 1

# dataset = np.array(dataset)
labels = np.array(labels)

print(labels.shape)

# np.save('./dataset_padded.npy', dataset)
np.save('./labels.npy', labels)
