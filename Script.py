import numpy as np
import package
import tensorflow as tf
import LSTM

# load
dataset = np.load('./dataset.npy')
labels = np.load('./labels.npy')
print('Dataset shape: {0}'.format(dataset.shape))
print('Labels shape: {0}'.format(labels.shape))
# split
train_data, validation_data, test_data = package.split(dataset)
train_labels, validation_labels, test_labels = package.split(labels)
Dataset = tf.data.Dataset.from_tensor_slices(dataset)
# model parameters
batch_size = 100
num_frames = 300
data = tf.placeholder(tf.float32, shape=[batch_size, num_frames, 1000])
label = tf.placeholder(tf.float32, shape=[batch_size, 51])
lstm = LSTM.LSTM(data, label, 128, 'normal', 1e-3,
                 'sgd', dropout=True, keep_prob=0.7)
# parameter initializer
initializer = tf.global_variables_initializer()
