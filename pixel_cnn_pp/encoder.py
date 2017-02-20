import tensorflow as tf
import numpy as np
import math
import time
from pixel_cnn_pp.nn import *

def lrelu(x, rate=0.1):
    # return tf.nn.relu(x)
    return tf.maximum(tf.minimum(x * rate, 0), x)

class ConvolutionalEncoder(object):
    def __init__(self, X, reg_type, latent_dim, z=None):
        '''
            This is the 6-layer architecture for Convolutional Autoencoder
            mentioned in the original paper:
            Stacked Convolutional Auto-Encoders for Hierarchical Feature Extraction

            Note that only the encoder part is implemented as PixelCNN is taken
            as the decoder.
        '''
        self.x = X
        conv1 = conv2d(X, 64, [4, 4], [2, 2], name='encoder_conv1')
        conv1 = lrelu(conv1)
        conv2 = conv2d(conv1, 128, [4, 4], [2, 2], name='encoder_conv2')
        conv2 = lrelu(conv2)
        conv3 = conv2d(conv2, 256, [4, 4], [2, 2], name='encoder_conv3')
        conv3 = lrelu(conv3)
        conv3 = tf.reshape(conv3, [-1, np.prod(conv3.get_shape().as_list()[1:])])
        fc1 = dense(conv3, 512, name='encoder_fc1')
        fc1 = lrelu(fc1)
        self.mean = dense(fc1, latent_dim, name='encoder_mean')
        self.stddev = tf.nn.sigmoid(dense(fc1, latent_dim, name='encoder_stddev'))
        self.stddev = tf.maximum(self.stddev, 0.01)
        self.pred = self.mean + tf.mul(self.stddev,
                                       tf.random_normal(tf.pack([tf.shape(X)[0], latent_dim])))

        if "elbo" in reg_type:
            self.reg_loss = tf.reduce_sum(-tf.log(self.stddev) + 0.5 * tf.square(self.stddev) +
                                           0.5 * tf.square(self.mean) - 0.5)
        elif "2norm" in reg_type:
            self.reg_loss = tf.reduce_sum(0.5 * tf.square(self.pred))
        elif "center" in reg_type:
            self.reg_loss = tf.reduce_sum(-tf.log(self.stddev) + 0.5 * tf.square(self.mean))
        elif "elbo0_1" in reg_type:
            self.reg_loss = 0.1 * tf.reduce_sum(-tf.log(self.stddev) + 0.5 * tf.square(self.stddev) +
                                           0.5 * tf.square(self.mean) - 0.5)
        elif "no_reg" in reg_type:
            self.reg_loss = 0.0 # Add something for stability
        else:
            print("Unknown regularization %s" % str(reg_type))
            exit(0)
        self.elbo_loss = tf.reduce_mean(-tf.log(self.stddev) + 0.5 * tf.square(self.stddev) +
                                        0.5 * tf.square(self.mean) - 0.5)

class ComputeLL:
    def __init__(self, latent_dim):
        self.mean = tf.placeholder(tf.float32, shape=(None, latent_dim))
        self.stddev = tf.placeholder(tf.float32, shape=(None, latent_dim))
        self.sample = tf.placeholder(tf.float32, shape=(None, latent_dim))
        mu = tf.reshape(self.mean, shape=tf.pack([tf.shape(self.mean)[0], 1, latent_dim]))
        mu = tf.tile(mu, tf.pack([1, tf.shape(self.sample)[0], 1]))
        sig = tf.reshape(self.stddev, shape=tf.pack([tf.shape(self.stddev)[0], 1, latent_dim]))
        sig = tf.tile(sig, tf.pack([1, tf.shape(self.sample)[0], 1]))
        z = tf.reshape(self.sample, shape=tf.pack([1, tf.shape(self.sample)[0], latent_dim]))
        z = tf.tile(z, tf.pack([tf.shape(self.mean)[0], 1, 1]))

        coeff = tf.div(1.0 / math.sqrt(2 * math.pi), sig)
        ll = coeff * tf.exp(-tf.div(tf.square(z - mu), 2 * tf.square(sig)))
        ll = tf.reduce_prod(ll, axis=2)
        self.prob = ll


def compute_mutual_information(data, args, sess, encoder_list, ll_compute):
    print("Evaluating Mutual Information")
    start_time = time.time()
    num_batch = 1000
    z_batch_cnt = 10  # This must divide num_batch
    dist_batch_cnt = 10
    assert num_batch % z_batch_cnt == 0
    assert num_batch % dist_batch_cnt == 0
    batch_size = args.batch_size * args.nr_gpu

    sample_batches = np.zeros((num_batch*batch_size, args.latent_dim))
    mean_batches = np.zeros((num_batch*batch_size, args.latent_dim))
    stddev_batches = np.zeros((num_batch*batch_size, args.latent_dim))

    for batch in range(num_batch):
        x = data.next(args.batch_size * args.nr_gpu) # manually retrieve exactly init_batch_size examples
        x = np.split(x, args.nr_gpu)
        feed_dict = {encoder_list[i].x: x[i] for i in range(args.nr_gpu)}

        result = sess.run([encoder.pred for encoder in encoder_list] +
                          [encoder.mean for encoder in encoder_list] +
                          [encoder.stddev for encoder in encoder_list], feed_dict=feed_dict)
        sample = np.concatenate(result[0:args.nr_gpu], 0)
        z_mean = np.concatenate(result[args.nr_gpu:args.nr_gpu*2], 0)
        z_stddev = np.concatenate(result[args.nr_gpu*2:], 0)
        sample_batches[batch*batch_size:(batch+1)*batch_size, :] = sample
        mean_batches[batch*batch_size:(batch+1)*batch_size, :] = z_mean
        stddev_batches[batch*batch_size:(batch+1)*batch_size, :] = z_stddev

    z_batch_size = batch_size * z_batch_cnt
    dist_batch_size = batch_size * dist_batch_cnt
    prob_array = np.zeros((num_batch*batch_size, num_batch*batch_size), dtype=np.float)
    for z_ind in range(num_batch // z_batch_cnt):
        for dist_ind in range(num_batch // dist_batch_cnt):
            mean = mean_batches[dist_ind*dist_batch_size:(dist_ind+1)*dist_batch_size, :]
            stddev = stddev_batches[dist_ind*dist_batch_size:(dist_ind+1)*dist_batch_size, :]
            sample = sample_batches[z_ind*z_batch_size:(z_ind+1)*z_batch_size, :]
            probs = sess.run(ll_compute.prob, feed_dict={ll_compute.mean: mean,
                                                         ll_compute.stddev: stddev,
                                                         ll_compute.sample: sample})
            prob_array[dist_ind*dist_batch_size:(dist_ind+1)*dist_batch_size, z_ind*z_batch_size:(z_ind+1)*z_batch_size] = probs
        # print()
    # print(np.sum(prob_array))
    marginal = np.sum(prob_array, axis=0)
    ratio = np.log(np.divide(np.diagonal(prob_array), marginal)) + np.log(num_batch*batch_size)
    mutual_info = np.mean(ratio)
    print("Mutual Information %f, time elapsed %fs" % (mutual_info, time.time() - start_time))
    return mutual_info