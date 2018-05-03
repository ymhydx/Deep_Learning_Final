import numpy as np
import tensorflow as tf
import vgg19


def baseline(mini_batch: np.ndarray, image_size):
    placeholder_size = [mini_batch.shape[0], image_size[0], image_size[1], 3]
    images = tf.placeholder("float", placeholder_size)
    vgg = vgg19.Vgg19()
    vgg.build(images)
    with tf.Session() as sess:
        feed_dict = {images: mini_batch}
        fc8 = sess.run(vgg.fc8, feed_dict=feed_dict)
        return fc8.shape
