"""
Trains a Pixel-CNN++ generative model on CIFAR-10 or Tiny ImageNet data.
Uses multiple GPUs, indicated by the flag --nr-gpu

Example usage:
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_double_cnn.py --nr_gpu 4
"""

import os
import sys
import time
import json
import argparse

import numpy as np
import tensorflow as tf
import scipy.misc

import pixel_cnn_pp.nn as nn
import pixel_cnn_pp.plotting as plotting
from pixel_cnn_pp.model import model_spec, model_spec_encoder
import data.cifar10_data as cifar10_data
import data.imagenet_data as imagenet_data
from pixel_cnn_pp.encoder import compute_mutual_information, ComputeLL
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
# data I/O
parser.add_argument('-i', '--data_dir', type=str, default='data', help='Location for the dataset')
parser.add_argument('-o', '--save_dir', type=str, default='elbo', help='Location for parameter checkpoints and samples')
parser.add_argument('-d', '--data_set', type=str, default='cifar', help='Can be either cifar|imagenet')
parser.add_argument('-t', '--save_interval', type=int, default=1, help='Every how many epochs to write checkpoint/samples?')
parser.add_argument('-r', '--load_params', dest='load_params', action='store_true', help='Restore training from previous model checkpoint?')
parser.add_argument('-name', '--name', type=str, default='elbo', help='Name of the network')
# model
parser.add_argument('-q', '--nr_resnet', type=int, default=5, help='Number of residual blocks per stage of the model')
parser.add_argument('-n', '--nr_filters', type=int, default=160, help='Number of filters to use across the model. Higher = larger model.')
parser.add_argument('-m', '--nr_logistic_mix', type=int, default=10, help='Number of logistic components in the mixture. Higher = more flexible model')
parser.add_argument('-z', '--resnet_nonlinearity', type=str, default='concat_elu', help='Which nonlinearity to use in the ResNet layers. One of "concat_elu", "elu", "relu" ')
parser.add_argument('-c', '--class_conditional', dest='class_conditional', action='store_true', help='Condition generative model on labels?')
parser.add_argument('-ae', '--use_autoencoder', dest='use_autoencoder', action='store_true', help='Use autoencoders?')
parser.add_argument('-reg', '--reg_type', type=str, default='elbo', help='Type of regularization to use for autoencoder')
parser.add_argument('-cs', '--chain_step', type=int, default=10, help='Steps to run Markov chain for sampling')
# optimization
parser.add_argument('-l', '--learning_rate', type=float, default=0.001, help='Base learning rate')
parser.add_argument('-e', '--lr_decay', type=float, default=0.999995, help='Learning rate decay, applied every step of the optimization')
parser.add_argument('-b', '--batch_size', type=int, default=12, help='Batch size during training per GPU')
parser.add_argument('-a', '--init_batch_size', type=int, default=80, help='How much data to use for data-dependent initialization.')
parser.add_argument('-p', '--dropout_p', type=float, default=0.5, help='Dropout strength (i.e. 1 - keep_prob). 0 = No dropout, higher = more dropout.')
parser.add_argument('-x', '--max_epochs', type=int, default=5000, help='How many epochs to run in total?')
parser.add_argument('-g', '--nr_gpu', type=int, default=2, help='How many GPUs to distribute the training across?')
parser.add_argument('-gid', '--gpu_id', type=str, default='', help='Which GPUs to use')
# evaluation
parser.add_argument('--polyak_decay', type=float, default=0.9995, help='Exponential decay rate of the sum of previous model iterates during Polyak averaging')
# reproducibility
parser.add_argument('-s', '--seed', type=int, default=1, help='Random seed to use')
args = parser.parse_args()
print('input args:\n', json.dumps(vars(args), indent=4, separators=(',',':'))) # pretty print args

# python train.py --use_autoencoder --save_dir=elbo --name=elbo --reg_type=elbo
# python train.py --use_autoencoder --save_dir=no_reg --name=no_reg --reg_type=no_reg
if args.gpu_id != "":
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

latent_dim = 20
args.latent_dim = latent_dim
# -----------------------------------------------------------------------------
# fix random seed for reproducibility
rng = np.random.RandomState(args.seed)
tf.set_random_seed(args.seed)

# initialize data loaders for train/test splits
if args.data_set == 'imagenet' and args.class_conditional:
    raise("We currently don't have labels for the small imagenet data set")
DataLoader = {'cifar':cifar10_data.DataLoader, 'imagenet':imagenet_data.DataLoader}[args.data_set]
train_data = DataLoader(args.data_dir, 'train', args.batch_size * args.nr_gpu, rng=rng, shuffle=True, return_labels=args.class_conditional)
test_data = DataLoader(args.data_dir, 'test', args.batch_size * args.nr_gpu, shuffle=False, return_labels=args.class_conditional)
obs_shape = train_data.get_observation_size() # e.g. a tuple (32,32,3)
assert len(obs_shape) == 3, 'assumed right now'

# data place holders
x_init = tf.placeholder(tf.float32, shape=(args.init_batch_size,) + obs_shape)
xs = [tf.placeholder(tf.float32, shape=(args.batch_size, ) + obs_shape) for i in range(args.nr_gpu)]
encoder_x_init = tf.placeholder(tf.float32, shape=(args.init_batch_size,) + obs_shape)
encoder_x = [tf.placeholder(tf.float32, shape=(args.batch_size, ) + obs_shape) for i in range(args.nr_gpu)]

# if the model is class-conditional we'll set up label placeholders + one-hot encodings 'h' to condition on
if args.class_conditional:
    num_labels = train_data.get_num_labels()
    y_init = tf.placeholder(tf.int32, shape=(args.init_batch_size,))
    h_init = tf.one_hot(y_init, num_labels)
    y_sample = np.split(np.mod(np.arange(args.batch_size*args.nr_gpu), num_labels), args.nr_gpu)
    h_sample = [tf.one_hot(tf.Variable(y_sample[i], trainable=False), num_labels) for i in range(args.nr_gpu)]
    ys = [tf.placeholder(tf.int32, shape=(args.batch_size,)) for i in range(args.nr_gpu)]
    hs = [tf.one_hot(ys[i], num_labels) for i in range(args.nr_gpu)]
elif args.use_autoencoder:
    # h_init = tf.placeholder(tf.float32, shape=(args.init_batch_size, latent_dim))
    h_sample = [tf.placeholder(tf.float32, shape=(args.batch_size, latent_dim)) for i in range(args.nr_gpu)]
else:
    h_init = None
    h_sample = [None] * args.nr_gpu
    hs = h_sample

# create the model
model_opt = { 'nr_resnet': args.nr_resnet, 'nr_filters': args.nr_filters, 'nr_logistic_mix': args.nr_logistic_mix, 'resnet_nonlinearity': args.resnet_nonlinearity }
model = tf.make_template('model', model_spec)
if args.use_autoencoder:
    encoder_opt = model_opt.copy()
    encoder_opt['reg_type'] = args.reg_type
    encoder_opt['latent_dim'] = latent_dim
    encoder_model = tf.make_template('encoder', model_spec_encoder)

# run once for data dependent initialization of parameters
if args.use_autoencoder:
    encoder = encoder_model(encoder_x_init, init=True, dropout_p=args.dropout_p, **encoder_opt)
    gen_par = model(x_init, encoder.pred, init=True, dropout_p=args.dropout_p, **model_opt)
else:
    gen_par = model(x_init, h_init, init=True, dropout_p=args.dropout_p, **model_opt)

# keep track of moving average
all_params = tf.trainable_variables()
ema = tf.train.ExponentialMovingAverage(decay=args.polyak_decay)
maintain_averages_op = tf.group(ema.apply(all_params))

# get loss gradients over multiple GPUs
grads = []
loss_gen = []
loss_gen_reg = []
loss_gen_elbo = []
loss_gen_test = []
for i in range(args.nr_gpu):
    with tf.device('/gpu:%d' % i):
        # train
        if args.use_autoencoder:
            encoder = encoder_model(encoder_x[i], ema=None, dropout_p=args.dropout_p, **encoder_opt)
            gen_par = model(xs[i], encoder.pred, ema=None, dropout_p=args.dropout_p, **model_opt)
            loss_gen_reg.append(encoder.reg_loss)
            loss_gen_elbo.append(encoder.elbo_loss)
        else:
            gen_par = model(xs[i], hs[i], ema=None, dropout_p=args.dropout_p, **model_opt)
        loss_gen.append(nn.discretized_mix_logistic_loss(xs[i], gen_par))
        # gradients
        if args.use_autoencoder:
            total_loss = loss_gen[i] + loss_gen_reg[i]
        else:
            total_loss = loss_gen[i]
        grads.append(tf.gradients(total_loss, all_params))
        # test
        if args.use_autoencoder:
            encoder = encoder_model(encoder_x[i], ema=ema, dropout_p=0., **encoder_opt)
            gen_par = model(xs[i], encoder.pred, ema=ema, dropout_p=0., **model_opt)
        else:
            gen_par = model(xs[i], hs[i], ema=ema, dropout_p=0., **model_opt)
        loss_gen_test.append(nn.discretized_mix_logistic_loss(xs[i], gen_par))

# add losses and gradients together and get training updates
tf_lr = tf.placeholder(tf.float32, shape=[])
with tf.device('/gpu:0'):
    for i in range(1,args.nr_gpu):
        loss_gen[0] += loss_gen[i]
        loss_gen_test[0] += loss_gen_test[i]
        if args.use_autoencoder:
            loss_gen_reg[0] += loss_gen_reg[i]
            loss_gen_elbo[0] += loss_gen_elbo[i]
        for j in range(len(grads[0])):
            grads[0][j] += grads[i][j]
    # training op
    tf.summary.scalar('ll_loss', loss_gen[0])
    if args.use_autoencoder:
        tf.summary.scalar('reg', loss_gen_reg[0])
        tf.summary.scalar('elbo', loss_gen_elbo[0])
    optimizer = tf.group(nn.adam_updates(all_params, grads[0], lr=tf_lr, mom1=0.95, mom2=0.9995), maintain_averages_op)

# convert loss to bits/dim
bits_per_dim = loss_gen[0]/(args.nr_gpu*np.log(2.)*np.prod(obs_shape)*args.batch_size)
bits_per_dim_test = loss_gen_test[0]/(args.nr_gpu*np.log(2.)*np.prod(obs_shape)*args.batch_size)
tf.summary.scalar('ll_bits_per_dim', bits_per_dim)

# sample from the model
new_x_gen = []
encoder_list = []
for i in range(args.nr_gpu):
    with tf.device('/gpu:%d' % i):
        if args.use_autoencoder:
            encoder = encoder_model(encoder_x[i], ema=ema, dropout_p=0, **encoder_opt)
            gen_par = model(xs[i], h_sample[i], ema=ema, dropout_p=0, **model_opt)
            encoder_list.append(encoder)
        else:
            gen_par = model(xs[i], h_sample[i], ema=ema, dropout_p=0, **model_opt)
        new_x_gen.append(nn.sample_from_discretized_mix_logistic(gen_par, args.nr_logistic_mix))
compute_ll = ComputeLL(latent_dim)

def sample_from_model(sess):
    x_gen = [np.zeros((args.batch_size,) + obs_shape, dtype=np.float32) for i in range(args.nr_gpu)]
    for yi in range(obs_shape[0]):
        for xi in range(obs_shape[1]):
            new_x_gen_np = sess.run(new_x_gen, {xs[i]: x_gen[i] for i in range(args.nr_gpu)})
            for i in range(args.nr_gpu):
                x_gen[i][:,yi,xi,:] = new_x_gen_np[i][:,yi,xi,:]
    return np.concatenate(x_gen, axis=0)

def sample_from_decoder_prior(sess):
    x_gen = [np.zeros((args.batch_size,) + obs_shape, dtype=np.float32) for i in range(args.nr_gpu)]
    latent_code = [np.random.normal(size=(args.batch_size, latent_dim)) for i in range(args.nr_gpu)]
    for yi in range(obs_shape[0]):
        for xi in range(obs_shape[1]):
            feed_dict = {xs[i]: x_gen[i] for i in range(args.nr_gpu)}
            feed_dict.update({h_sample[i]: latent_code[i] for i in range(args.nr_gpu)})
            new_x_gen_np = sess.run(new_x_gen, feed_dict)
            for i in range(args.nr_gpu):
                x_gen[i][:,yi,xi,:] = new_x_gen_np[i][:,yi,xi,:]
    return np.concatenate(x_gen, axis=0)

def sample_from_markov_chain(sess, initial=None):
    history = []
    if initial is None:
        encoder_current = [np.random.uniform(0.0, 1.0, (args.batch_size,) + obs_shape) for i in range(args.nr_gpu)]
    else:
        encoder_current = np.split(initial, args.nr_gpu)
    latent_op = [encoder.pred for encoder in encoder_list]
    num_steps = args.chain_step
    history.append(np.concatenate(encoder_current, axis=0))

    for step in range(num_steps):
        start_time = time.time()
        feed_dict = {encoder_x[i]: encoder_current[i] for i in range(args.nr_gpu)}
        latent_code = sess.run(latent_op, feed_dict)

        x_gen = [np.zeros((args.batch_size,) + obs_shape, dtype=np.float32) for i in range(args.nr_gpu)]
        for yi in range(obs_shape[0]):
            for xi in range(obs_shape[1]):
                feed_dict = {xs[i]: x_gen[i] for i in range(args.nr_gpu)}
                feed_dict.update({h_sample[i]: latent_code[i] for i in range(args.nr_gpu)})
                new_x_gen_np = sess.run(new_x_gen, feed_dict)
                for i in range(args.nr_gpu):
                    x_gen[i][:,yi,xi,:] = new_x_gen_np[i][:,yi,xi,:]
        history.append(np.concatenate(x_gen, axis=0))
        encoder_current = x_gen
        print("%d (%fs)" % (step, time.time() - start_time))
        sys.stdout.flush()
    return history

def plot_markov_chain(history):
    canvas = np.zeros((args.nr_gpu*args.batch_size*obs_shape[0], len(history)*obs_shape[1], obs_shape[2]))
    for i in range(args.nr_gpu*args.batch_size):
        for j in range(len(history)):
            canvas[i*obs_shape[0]:(i+1)*obs_shape[0], j*obs_shape[1]:(j+1)*obs_shape[1], :] = history[j][i]
    return canvas

# init & save
initializer = tf.initialize_all_variables()
saver = tf.train.Saver()
all_summary = tf.summary.merge_all()
writer = tf.summary.FileWriter(logdir=args.save_dir)
file_logger = open(os.path.join(args.save_dir, 'train_log'), 'w')
# turn numpy inputs into feed_dict for use with tensorflow
def make_feed_dict(data, init=False):
    if type(data) is tuple:
        x,y = data
    else:
        x = data
        y = None
    x = np.cast[np.float32]((x - 127.5) / 127.5) # input to pixelCNN is scaled from uint8 [0,255] to float in range [-1,1]
    if init:
        feed_dict = {x_init: x}
        if args.use_autoencoder:
            feed_dict.update({encoder_x_init: x})
        if y is not None:
            feed_dict.update({y_init: y})
    else:
        x = np.split(x, args.nr_gpu)
        feed_dict = {xs[i]: x[i] for i in range(args.nr_gpu)}
        if args.use_autoencoder:
            feed_dict.update({encoder_x[i]: x[i] for i in range(args.nr_gpu)})
        if y is not None:
            y = np.split(y, args.nr_gpu)
            feed_dict.update({ys[i]: y[i] for i in range(args.nr_gpu)})
    return feed_dict

# //////////// perform training //////////////
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
print('starting training')
test_bpd = []
lr = args.learning_rate
global_step = 0

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9, allow_growth=True)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)) as sess:
    for epoch in range(args.max_epochs):
        # init
        if epoch == 0:
            feed_dict = make_feed_dict(train_data.next(args.init_batch_size), init=True) # manually retrieve exactly init_batch_size examples
            train_data.reset()  # rewind the iterator back to 0 to do one full epoch
            sess.run(initializer, feed_dict)
            print('initializing the model...')
            if args.load_params:
                ckpt_file = args.save_dir + '/params_' + args.data_set + '.ckpt'
                print('restoring parameters from', ckpt_file)
                saver.restore(sess, ckpt_file)

        # Compute mutual information
        file_logger.write("%d " % epoch)
        if args.use_autoencoder:
            mutual_info = compute_mutual_information(data=train_data, args=args, sess=sess, encoder_list=encoder_list, ll_compute=compute_ll)
            train_data.reset()
            file_logger.write("%f " % mutual_info)
            file_logger.flush()

        # generate samples from the model
        if args.use_autoencoder and epoch % 20 == 0:
            print("Generating MC")
            start_time = time.time()
            initial = np.random.uniform(0.0, 1.0, (args.batch_size * args.nr_gpu,) + obs_shape)
            for mc_step in range(100):
                sample_history = sample_from_markov_chain(sess, initial)
                initial = sample_history[-1]
                sample_plot = plot_markov_chain(sample_history)
                scipy.misc.imsave(os.path.join(args.save_dir, '%s_mc%d.png' % (args.data_set, mc_step)), sample_plot)
            print("Finished, time elapsed %fs" % (time.time() - start_time))
            exit(0)

        # generate samples from the model
        if epoch % 2 == 0:
            print("Generating samples")
            start_time = time.time()
            if args.use_autoencoder:
                sample_x = sample_from_decoder_prior(sess)
            else:
                sample_x = sample_from_model(sess)
            img_tile = plotting.img_tile(sample_x[:int(np.floor(np.sqrt(args.batch_size * args.nr_gpu)) ** 2)],
                                         aspect_ratio=1.0, border_color=1.0, stretch=True)
            img = plotting.plot_img(img_tile, title=args.data_set + ' samples')
            plotting.plt.savefig(os.path.join(args.save_dir, '%s_sample%d.png' % (args.data_set, epoch)))
            plotting.plt.close('all')
            print("Finished, time elapsed %fs" % (time.time() - start_time))

        begin = time.time()
        # train for one epoch
        train_losses = []
        batch_c = 10
        for d in train_data:
            feed_dict = make_feed_dict(d)
            # forward/backward/update model on each gpu
            lr *= args.lr_decay
            feed_dict.update({ tf_lr: lr })
            l, _, summaries = sess.run([bits_per_dim, optimizer, all_summary], feed_dict)
            train_losses.append(l)
            if global_step % 5 == 0:
                writer.add_summary(summaries, global_step)
            global_step += 1
        train_loss_gen = np.mean(train_losses)

        # compute likelihood over test data
        test_losses = []
        for d in test_data:
            feed_dict = make_feed_dict(d)
            l = sess.run(bits_per_dim_test, feed_dict)
            test_losses.append(l)
        test_loss_gen = np.mean(test_losses)
        test_bpd.append(test_loss_gen)
        file_logger.write("%f\n" % test_loss_gen)

        # log progress to console
        print("Iteration %d, time = %ds, train bits_per_dim = %.4f, test bits_per_dim = %.4f" % (epoch, time.time()-begin, train_loss_gen, test_loss_gen))
        sys.stdout.flush()

        if epoch % args.save_interval == 0:
            # save params
            saver.save(sess, args.save_dir + '/params_' + args.data_set + '.ckpt')
            np.savez(args.save_dir + '/test_bpd_' + args.data_set + '.npz', test_bpd=np.array(test_bpd))
