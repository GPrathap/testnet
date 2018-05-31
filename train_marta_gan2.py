import os
import sys
import pprint
import time

from Utils import save_images
from convert_to_tf_record import DataConvertor
from network import Neotx
import numpy as np
pp = pprint.PrettyPrinter()

from Config import *

def main(_):
    pp.pprint(flags.FLAGS.__flags)

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

    z_dim = 100

    # with tf.device("/gpu:0"): # <-- if you have a GPU machine
    z = tf.placeholder(tf.float32, [FLAGS.batch_size, z_dim], name='z_noise')

    data_convotor = DataConvertor(FLAGS.image_size, FLAGS.dataset_name,
                                  FLAGS.dataset_storage_location, FLAGS.c_dim)

    next_batch, iterator = data_convotor.provide_data(FLAGS.batch_size,  'train')
    real_images = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.output_size,
                                              FLAGS.output_size, FLAGS.c_dim], name='real_images')
    neoxt = Neotx()
    # z --> generator for training
    net_g, g_logits = neoxt.generator(z, is_train=True, reuse=tf.AUTO_REUSE)
    # generated fake images --> discriminator
    net_d, d_logits, feature_fake = neoxt.discriminator(net_g, is_train=True, reuse=tf.AUTO_REUSE)
    # real images --> discriminator
    net_d2, d2_logits, feature_real = neoxt.discriminator(real_images, is_train=True, reuse=True)
    # sample_z --> generator for evaluation, set is_train to False

    net_g2, g2_logits = neoxt.generator(z, is_train=False, reuse=True)
    net_d3, d3_logits, _ = neoxt.discriminator(real_images, is_train=False, reuse=True)

    # cost for updating discriminator and generator
    # discriminator: real images are labelled as 1
    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d2_logits,
                                                                         labels=tf.ones_like(d2_logits)))
    # real == 1
    # discriminator: images from generator (fake) are labelled as 0
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits,
                                                                         labels=tf.zeros_like(d_logits)))
    # fake == 0
    d_loss = d_loss_real + d_loss_fake
    # generator: try to make the the fake images look real (1)
    g_loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits,
                                                                     labels=tf.ones_like(d_logits)))
    g_loss2 = tf.reduce_mean(tf.nn.l2_loss(feature_real-feature_fake))/(FLAGS.image_size*FLAGS.image_size)
    g_loss = g_loss1 + g_loss2

    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = 0.1
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                               1000, FLAGS.learning_rate, staircase=True)

    d_optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=FLAGS.beta1)
    g_optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=FLAGS.beta1)

    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_update_ops):
        g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator/*")
        d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator/*")
        d_optim = d_optimizer.minimize(d_loss, var_list=d_vars)
        g_optim = g_optimizer.minimize(g_loss, var_list=g_vars)

    saver = tf.train.Saver(max_to_keep=100)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        total_parameters = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
        print("Total number of parameters: "+ str(total_parameters))
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("[*] Loading checkpoints ...")
        else:
            print("[*] Loading checkpoints failed ...")
        sample_seed = np.random.uniform(low=-1, high=1, size=(FLAGS.batch_size, z_dim)).astype(np.float32)

        for epoch in range(FLAGS.epoch):
            iter_counter = 0
            batch_images_for_testing = []
            sess.run(iterator.initializer)
            while True:
                try:
                    batch_images = sess.run([next_batch])
                    batch_images = np.array(batch_images[0][0], dtype=np.float32)/127.5-1
                    batch_z = np.random.uniform(low=-1, high=1, size=(FLAGS.batch_size, z_dim))\
                        .astype(np.float32)
                    start_time = time.time()

                    for _ in range(1):
                        errD, _ = sess.run([d_loss, d_optim], feed_dict={z: batch_z
                            , real_images: batch_images})
                    for _ in range(2):
                        errG, _ = sess.run([g_loss, g_optim], feed_dict={z: batch_z
                            , real_images: batch_images})
                    print("Epoch: [%2d/%2d] [%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                          % (epoch, FLAGS.epoch, iter_counter,
                             time.time() - start_time, errD, errG))
                    sys.stdout.flush()
                    iter_counter += 1
                    if iter_counter == 1:
                        batch_images_for_testing = batch_images

                except tf.errors.OutOfRangeError:
                    break

            if np.mod(epoch, 1) == 0:
                img, errG = sess.run([net_g2, g_loss],
                                     feed_dict={z : sample_seed, real_images: batch_images_for_testing})
                D, D_, errD = sess.run([net_d3, net_d3, d_loss_real],
                                       feed_dict={real_images: batch_images_for_testing})

                save_images(img, [8, 8], '{}/train_{:02d}.png'.format(FLAGS.sample_dir, epoch))
                print("[Sample] d_loss: %.8f, g_loss: %.8f" % (errD, errG))
                sys.stdout.flush()

            if np.mod(epoch, 5) == 0:
                print("[*] Saving checkpoints...")
                save_path = saver.save(sess, FLAGS.checkpoint_dir + '/model', global_step=epoch)
                print("Model saved in path: %s" % save_path)
                print("[*] Saving checkpoints SUCCESS!")

def describe_network(sess):
     variables_names = [v.name for v in tf.trainable_variables()]
     values = sess.run(variables_names)
     for k, v in zip(variables_names, values):
        print("Variable: ", k)
        print("Shape: ", v.shape)

if __name__ == '__main__':
    tf.app.run()
