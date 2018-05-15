import numpy as np
import package
import tensorflow as tf
import LSTM

dataset = np.load('./dataset.npy')
labels = np.load('./labels.npy')
print(dataset[0].shape)
print(dataset.shape)
print(labels.shape)

train_data, validation_data, test_data = package.split(dataset)
train_labels, validation_labels, test_labels = package.split(labels)

batch_size = 100
data = tf.placeholder(tf.float32, shape=[batch_size, 30, 1000])
label = tf.placeholder(tf.float32, shape=[batch_size, 51])
lstm = LSTM.LSTM(data, label, 16, 'normal', 0.03, 'sgd')

initializer = tf.global_variables_initializer()

epochs = 100000
with tf.Session() as sess:
    sess.run(initializer)
    while epochs > 0:
        x, y = package.get(train_data, train_labels, batch_size)
        x_feed = []
        y_feed = y
        for i in range(x.shape[0]):
            x_feed.append(package.subsample(x[i], 30))
        x_feed = np.array(x_feed)
        pred, loss, _ = sess.run([lstm.pred, lstm.loss, lstm.optimizer], feed_dict={data: x_feed, label: y_feed})
        # if not epochs % 100:
        #     validation_feed = []
        #     for i in range(validation_data.shape[0]):
        #         validation_feed.append(package.maximum_length_padding(validation_data[i]))
        #     validation_feed = np.array(validation_feed)
        if not epochs % 1000:
            print(loss)
        epochs -= 1
