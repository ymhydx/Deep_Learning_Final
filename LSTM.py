import tensorflow as tf
import numpy as np
import Padding


class LSTM:
    def __init__(self, input_vgg: tf.placeholder, labels: tf.placeholder, n_hidden, init_type, learning_rate, opt_type,
                 dropout=False, keep_prob=1):
        self.output, self.optimizer = self.build(input_vgg, labels, n_hidden, init_type, learning_rate, opt_type,
                                                 dropout, keep_prob)

    def build(self, input_vgg: tf.placeholder, labels: tf.placeholder, n_hidden, init_type, learning_rate, opt_type,
              dropout=False, keep_prob=1):
        num_class = 51
        # input_vgg = tf.placeholder(tf.float32, shape=[None, num_frames, 1000])  # input from vgg19
        cell = tf.contrib.rnn.LSTMCell(n_hidden)
        if dropout:
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
        output_rnn, state = tf.nn.dynamic_rnn(cell, input_vgg, dtype=tf.float32)
        output_rnn = tf.transpose(output_rnn, [1, 0, 2])
        output_last = tf.gather(output_rnn, int(output_rnn.shape[0]) - 1)
        if init_type == 'normal':
            W = tf.Variable(tf.truncated_normal([n_hidden, num_class], stddev=0.1))
        elif init_type == 'xavier':
            W = tf.get_variable(name='W', shape=[n_hidden, num_class],
                                initializer=tf.contrib.layers.xavier_initializer())
        b = tf.Variable(tf.constant(0.1, shape=[num_class]))
        output = tf.matmul(output_last, W) + b

        pred = tf.nn.softmax(output)
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=labels))
        if opt_type == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=cross_entropy)
        elif opt_type == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss=cross_entropy)
        return output, optimizer


test_video = np.load('./test_video.npy')
batch = [Padding.maximum_length_padding(test_video)]
batch = np.array(batch)
test_label = np.zeros([1, 51])
print(batch.shape)
print(test_label.shape)
# # demo
batch_size = 1
input_vgg = tf.placeholder(tf.float32, shape=[batch_size, 1062, 1000])  # shape=[batch_size,num_frames,features]
labels = tf.placeholder(tf.float32, shape=[batch_size, 51])  # shape=[batch_size, labels]
lstm = LSTM(input_vgg, labels, 16, 'normal', 0.1, 'adam')
# output, optimizer = lstm.output, lstm.optimizer
# print(output.shape)

initializer = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(initializer)
    output, _ = sess.run([lstm.output, lstm.optimizer], feed_dict={input_vgg: batch, labels: test_label})
    print(output)
