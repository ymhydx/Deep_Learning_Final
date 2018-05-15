import package
import numpy as np

dataset = np.load('./dataset.npy')
temp = dataset[19]
print(temp.shape)
temp = package.subsample(temp, 30)
print(temp.shape)
