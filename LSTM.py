import tensorflow as tf
import numpy as np


class lstm:
    def __init__(self, batch: np.ndarray, batch_label, n_hidden, n_frames, learning_rate, opt_type, init_type,
                 dropout=False,
                 keep_prob=1):
        self.batch_size = batch.shape[0]
        self.n_hidden = n_hidden
        self.n_frames = n_frames
        self.learning_rate = learning_rate
        self.opt_type = opt_type
        self.init_type = init_type
        self.dropout = dropout
        self.keep_prob = keep_prob
        self.num_class = 51

    def build(self):
        input = tf.placeholder(tf.float32, shape=[self.n_frames, 1000, 1])
        cell = tf.contrib.rnn.LSTMCell(self.n_hidden)
        if self.dropout:
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
        output_rnn, state = tf.nn.dynamic_rnn(cell, input, dtype=tf.float32)
        output_rnn = tf.transpose(output_rnn, [1, 0, 2])
        output_last = tf.gather(output_rnn, int(output_rnn.shape[0]) - 1)
        if self.init_type == 'normal':
            W = tf.Variable(tf.truncated_normal([self.n_hidden, self.num_class], stddev=0.1))
        elif self.init_type == 'xavier':
            W = tf.get_variable(name='W', shape=[self.n_hidden, self.num_class],
                                initializer=tf.contrib.layers.xavier_initializer())

        b = tf.Variable(tf.constant(0.1, shape=[self.num_class]))
        output = tf.matmul(output_last, W) + b
