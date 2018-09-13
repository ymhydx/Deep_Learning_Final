import numpy as np
import package
import tensorflow as tf
import LSTM

# tensorflow configuration
tf.reset_default_graph()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# some parameters
batch_size = 200
num_frames = 70
num_epochs=25

# load
train_features=np.load('./train_features.npy')
train_labels=np.load('./train_labels.npy')
validation_features=np.load('./validation_features.npy')
validation_labels=np.load('./validation_labels.npy')
test_features=np.load('./test_features.npy')
test_labels=np.load('./test_labels.npy')

# size check
print('Train feature shape: {0}'.format(train_features.shape))
print('Train labels shape: {0}'.format(train_labels.shape))
print('Validation feature shape: {0}'.format(validation_features.shape))
print('Validation labels shape: {0}'.format(validation_labels.shape))
print('Test feature shape: {0}'.format(test_features.shape))
print('Test labels shape: {0}'.format(test_labels.shape))
# build tensor flow dataset
train_features_holder = tf.placeholder(
    train_features.dtype, train_features.shape)
train_labels_holder = tf.placeholder(train_labels.dtype, train_labels.shape)

# validation_features_holder = tf.placeholder(
#     validation_features.dtype, validation_features.shape)
# validation_labels_holder = tf.placeholder(validation_labels.dtype, validation_labels.shape)

Data = tf.data.Dataset.from_tensor_slices(
    (train_features_holder, train_labels_holder))
Batches=Data.batch(batch_size)

# Validation= tf.data.Dataset.from_tensor_slices(
#     (validation_features_holder, validation_labels_holder))
# Batch_valid=Validation.batch(5)

# model parameters
data = tf.placeholder(tf.float32, shape=[None, num_frames, 1000])
label = tf.placeholder(tf.float32, shape=[None, 51])
lstm = LSTM.LSTM(data, label, 64, 'normal', 1e-3,
                 'adam', dropout=True, keep_prob=0.7)

# parameter initializer
initializer = tf.global_variables_initializer()

# initializable iterator
iterator = Batches.make_initializable_iterator()
val=iterator.get_next()

# iterator_valid=Batch_valid.make_initializable_iterator()
# val_valid=iterator_valid.get_next()

# train model
with tf.Session(config=config) as sess:
    sess.run(initializer)
    for epoch in range(num_epochs):
        sess.run(iterator.initializer,feed_dict={train_features_holder: train_features, train_labels_holder:train_labels})
        loss_avg=0
        loss_total=0
        count=0
        try:
            while(True):
                feature_feed,label_feed=sess.run(val)
                pred, loss, _=sess.run([lstm.pred,lstm.loss,lstm.optimizer],feed_dict={data:feature_feed,label:label_feed})
                count+=1
                loss_total+=loss
        except:
            loss_avg=loss_total/count
            print('Epoch {0}, average loss: {1}'.format(epoch, loss_avg))

#             sess.run(iterator_valid.initializer,feed_dict={validation_features_holder: validation_features, validation_labels_holder:validation_labels})
#             feature_feed_valid,label_feed_valid=sess.run(val_valid)
            pred=sess.run(lstm.pred,feed_dict={data:validation_features,label:validation_labels})
            print(package.accuracy(package.prob2one_hot(pred),validation_labels))