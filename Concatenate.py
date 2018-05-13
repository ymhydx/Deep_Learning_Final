import Package
import os
import numpy as np

one_hot = Package.one_hot_encoder(51)
folders = os.listdir('./after_vgg19')
for folder in folders:
    files = os.listdir('./after_vgg19' + '/' + folder)
    for file in files:
        features = np.load('./after_vgg19' + '/' + folder + '/' + file)
        
