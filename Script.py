import numpy as np
import package
import tensorflow as tf
import LSTM

# load
features = np.load('./features.npy')
labels = np.load('./labels.npy')
print('Features shape: {0}'.format(features.shape))
print('Labels shape: {0}'.format(labels.shape))

# split
train_features, validation_features, test_features = package.split(features)
train_labels, validation_labels, test_labels = package.split(labels)

# build tensor flow dataset
train_features_holder = tf.placeholder(
    train_features.dtype, train_features.shape)
train_labels_holder = tf.placeholder(train_labels.dtype, train_labels.shape)
Data = tf.data.Dataset.from_tensor_slices(
    (train_features_holder, train_labels_holder))

# model parameters
batch_size = 100
num_frames = 70
data = tf.placeholder(tf.float32, shape=[batch_size, num_frames, 1000])
label = tf.placeholder(tf.float32, shape=[batch_size, 51])
lstm = LSTM.LSTM(data, label, 64, 'normal', 1e-3,
                 'sgd', dropout=True, keep_prob=0.7)

# parameter initializer
initializer = tf.global_variables_initializer()

# initializable iterator
iterator = Data.make_initializable_iterator()

# train model
with tf.Session as sess:
    sess.run(initializer, iterator.initializer, feed_dict={train_features_holder: train_features, train_labels_holder=train_labels})
