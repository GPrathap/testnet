import numpy as np
import tensorflow as tf
from utils import *


def conditional_generator_simplified_api(inputs, condition, batch_size, is_train=True, reuse=False):
    conditinal_input = tf.concat([inputs, condition], 0)
    return generator_simplified_api(conditinal_input, batch_size, is_train, reuse)


def batch_normalization_layer(layer, gamma_inijjt, is_training):
    gamma_init = tf.random_normal_initializer(1., 0.02)
    layer = tf.layers.batch_normalization(layer, epsilon=1e-12, gamma_initializer=gamma_init,
                                          training=is_training)
    return tf.nn.leaky_relu(layer, 0.2)
    #return layer

def generator_simplified_api(inputs, batch_size, is_train=True, reuse=False):
    gamma_init = tf.random_normal_initializer(1., 0.02)

    with tf.variable_scope("generator", reuse=reuse):

        depth_of_h0 = 256
        net_h01 = tf.layers.dense( inputs, depth_of_h0, activation=tf.identity)
        net_h01 = tf.reshape(net_h01, shape=(-1, 1, 1, depth_of_h0))
        net_h01 = batch_normalization_layer(net_h01, gamma_init, is_train)

        net_h02 = tf.layers.dense(inputs, depth_of_h0, activation=tf.identity)
        net_h02 = tf.reshape(net_h02, shape=(-1, 1, 1, depth_of_h0))
        net_h02 = batch_normalization_layer(net_h02, gamma_init, is_train)

        net_h03 = tf.layers.dense(inputs, depth_of_h0, activation=tf.identity)
        net_h03 = tf.reshape(net_h03, shape=(-1, 1, 1, depth_of_h0))
        net_h03 = batch_normalization_layer(net_h03, gamma_init, is_train)

        depth_of_h1 = int(depth_of_h0/2)
        net_h11 = tf.layers.conv2d_transpose(net_h01, depth_of_h1, [1, 1], strides=(2,2), padding='SAME', activation=None)
        net_h11 = batch_normalization_layer(net_h11, gamma_init, is_train)
        net_h12 = tf.layers.conv2d_transpose(net_h02, depth_of_h1, [3, 3], strides=(2, 2), padding='SAME', activation=None)
        net_h12 = batch_normalization_layer(net_h12, gamma_init, is_train)
        net_h13 = tf.layers.conv2d_transpose(net_h03, depth_of_h1, [5, 5], strides=(2, 2), padding='SAME', activation=None)
        net_h13 = batch_normalization_layer(net_h13, gamma_init, is_train)

        depth_of_h2 = int(depth_of_h1 / 2)
        net_h21 = tf.layers.conv2d_transpose(net_h11, depth_of_h2, [1, 1], strides=(2,2), padding='SAME', activation=None)
        net_h21 = batch_normalization_layer(net_h21, gamma_init, is_train)
        net_h22 = tf.layers.conv2d_transpose(net_h12, depth_of_h2, [3, 3], strides=(2, 2), padding='SAME', activation=None)
        net_h22 = batch_normalization_layer(net_h22, gamma_init, is_train)
        net_h23 = tf.layers.conv2d_transpose(net_h13, depth_of_h2, [5, 5], strides=(2, 2), padding='SAME', activation=None)
        net_h23 = batch_normalization_layer(net_h23, gamma_init, is_train)

        depth_of_h3 = int(depth_of_h2 / 2)
        net_h31 = tf.layers.conv2d_transpose(net_h21, depth_of_h3, [1, 1], strides=(2,2), padding='SAME', activation=None)
        net_h31 = batch_normalization_layer(net_h31, gamma_init, is_train)
        net_h32 = tf.layers.conv2d_transpose(net_h22, depth_of_h3, [3, 3], strides=(2, 2), padding='SAME', activation=None)
        net_h32 = batch_normalization_layer(net_h32, gamma_init, is_train)
        net_h33 = tf.layers.conv2d_transpose(net_h23, depth_of_h3, [5, 5], strides=(2, 2), padding='SAME', activation=None)
        net_h33 = batch_normalization_layer(net_h33, gamma_init, is_train)

        depth_of_h4 = int(depth_of_h3 / 2)
        net_h41 = tf.layers.conv2d_transpose(net_h31, depth_of_h4, [1, 1], strides=(2,2), padding='SAME', activation=None)
        net_h41 = batch_normalization_layer(net_h41, gamma_init, is_train)
        net_h42 = tf.layers.conv2d_transpose(net_h32, depth_of_h4, [3,3], strides=(2, 2), padding='SAME', activation=None)
        net_h42 = batch_normalization_layer(net_h42, gamma_init, is_train)
        net_h43 = tf.layers.conv2d_transpose(net_h33, depth_of_h4, [5,5], strides=(2, 2), padding='SAME', activation=None)
        net_h43 = batch_normalization_layer(net_h43, gamma_init, is_train)

        depth_of_h5 = int(depth_of_h4/2)
        net_h51 = tf.layers.conv2d_transpose(net_h41, depth_of_h5, [1,1], strides=(2, 2), padding='SAME', activation=None)
        net_h51 = batch_normalization_layer(net_h51, gamma_init, is_train)
        net_h52 = tf.layers.conv2d_transpose(net_h42, depth_of_h5, [3, 3], strides=(2, 2), padding='SAME', activation=None)
        net_h52 = batch_normalization_layer(net_h52, gamma_init, is_train)
        net_h53 = tf.layers.conv2d_transpose(net_h43, depth_of_h5, [5, 5], strides=(2, 2), padding='SAME', activation=None)
        net_h53 = batch_normalization_layer(net_h53, gamma_init, is_train)

        depth_of_h6 = int(depth_of_h5 / 2)
        net_h61 = tf.layers.conv2d_transpose(net_h51, depth_of_h6, [1, 1], strides=(2, 2), padding='SAME',
                                             activation=None)
        #net_h61 = batch_normalization_layer(net_h61, gamma_init, is_train)
        net_h62 = tf.layers.conv2d_transpose(net_h52, depth_of_h6, [3, 3], strides=(2, 2), padding='SAME',
                                             activation=None)
        #net_h62 = batch_normalization_layer(net_h62, gamma_init, is_train)
        net_h63 = tf.layers.conv2d_transpose(net_h53, depth_of_h6, [5, 5], strides=(2, 2), padding='SAME',
                                             activation=None)
        #net_h63 = batch_normalization_layer(net_h63, gamma_init, is_train)

        net_h71 = tf.concat(axis=3, values=[net_h61, net_h62, net_h63])
        net_h72 = tf.layers.conv2d_transpose(net_h71, 6, [1, 1], strides=(1, 1), padding='SAME',
                                             activation=tf.identity)
        net_h73 = tf.layers.conv2d_transpose(net_h72, 3, [1, 1], strides=(1, 1), padding='SAME',
                                             activation=tf.identity)

        logits = net_h73
        net_h7 = tf.nn.tanh(net_h73)
    return net_h7, logits

def conditional_discriminator_simplified_api(inputs, condition, is_train=True, reuse=False):
    conditinal_input = tf.concat([inputs, condition], 0)
    return discriminator_simplified_api(conditinal_input, is_train, reuse)

def discriminator_simplified_api(inputs, is_train=True, reuse=False):
    depth_of_h0 = 16
    gamma_init = tf.random_normal_initializer(1., 0.02)

    with tf.variable_scope("discriminator", reuse=reuse):

        net_h01 = tf.layers.conv2d(inputs, depth_of_h0, [1, 1], strides=(2,2), padding='SAME')
        net_h02 = tf.layers.conv2d(inputs, depth_of_h0, [3, 3], strides=(2,2), padding='SAME')
        net_h03 = tf.layers.conv2d(inputs, depth_of_h0, [5, 5], strides=(2,2), padding='SAME')

        depth_of_h1 = depth_of_h0*2
        net_h11 = tf.layers.conv2d(net_h01, depth_of_h1, [1, 1], strides=(2,2), padding='SAME', activation=None)
        net_h11 = batch_normalization_layer(net_h11, gamma_init, is_train)
        net_h12 = tf.layers.conv2d(net_h02, depth_of_h1, [3, 3], strides=(2, 2), padding='SAME', activation=None)
        net_h12 = batch_normalization_layer(net_h12, gamma_init, is_train)
        net_h13 = tf.layers.conv2d(net_h03, depth_of_h1, [5, 5], strides=(2, 2), padding='SAME',activation=None)
        net_h13 = batch_normalization_layer(net_h13, gamma_init, is_train)

        depth_of_h2 = depth_of_h1*2
        net_h21 = tf.layers.conv2d(net_h11, depth_of_h2, [1, 1], strides=(2,2), padding='SAME', activation=None)
        net_h21 = batch_normalization_layer(net_h21, gamma_init, is_train)
        net_h22 = tf.layers.conv2d(net_h12, depth_of_h2, [3, 3], strides=(2, 2), padding='SAME',activation=None)
        net_h22 = batch_normalization_layer(net_h22, gamma_init, is_train)
        net_h23 = tf.layers.conv2d(net_h13, depth_of_h2, [5, 5], strides=(2, 2), padding='SAME',activation=None)
        net_h23 = batch_normalization_layer(net_h23, gamma_init, is_train)

        depth_of_h3 = depth_of_h2*2
        net_h31 = tf.layers.conv2d(net_h21, depth_of_h3, [1, 1], strides=(2,2), padding='SAME',activation=None)
        net_h31 = batch_normalization_layer(net_h31, gamma_init, is_train)
        net_h32 = tf.layers.conv2d(net_h22, depth_of_h3, [2, 2], strides=(2, 2), padding='SAME',activation=None)
        net_h32 = batch_normalization_layer(net_h32, gamma_init, is_train)
        net_h33 = tf.layers.conv2d(net_h23, depth_of_h3, [3, 3], strides=(2, 2), padding='SAME', activation=None)
        net_h33 = batch_normalization_layer(net_h33, gamma_init, is_train)

        global_max11 = tf.layers.flatten(net_h31)
        global_max12 = tf.layers.flatten(net_h32)
        global_max13 = tf.layers.flatten(net_h33)

        depth_of_h4 = depth_of_h3 * 2
        net_h41 = tf.layers.conv2d(net_h31, depth_of_h4, [1, 1], strides=(2,2), padding='SAME',activation=None)
        net_h41 = batch_normalization_layer(net_h41, gamma_init, is_train)
        net_h42 = tf.layers.conv2d(net_h32, depth_of_h4, [3, 3], strides=(2, 2), padding='SAME',activation=None)
        net_h42 = batch_normalization_layer(net_h42, gamma_init, is_train)
        net_h43 = tf.layers.conv2d(net_h33, depth_of_h4, [5, 5], strides=(2, 2), padding='SAME',activation=None)
        net_h43 = batch_normalization_layer(net_h43, gamma_init, is_train)

        global_max21 = tf.layers.flatten(net_h41)
        global_max22 = tf.layers.flatten(net_h42)
        global_max23 = tf.layers.flatten(net_h43)

        depth_of_h5 = depth_of_h4 * 2
        net_h51 = tf.layers.conv2d(net_h41, depth_of_h5, [1, 1], strides=(2,2), padding='SAME',activation=None)
        net_h51 = batch_normalization_layer(net_h51, gamma_init, is_train)
        net_h52 = tf.layers.conv2d(net_h42, depth_of_h5, [3, 3], strides=(2, 2), padding='SAME',activation=None)
        net_h52 = batch_normalization_layer(net_h52, gamma_init, is_train)
        net_h53 = tf.layers.conv2d(net_h43, depth_of_h5, [5, 5], strides=(2, 2), padding='SAME',activation=None)
        net_h53 = batch_normalization_layer(net_h53, gamma_init, is_train)

        global_max31 = tf.layers.flatten(net_h51)
        global_max32 = tf.layers.flatten(net_h52)
        global_max33 = tf.layers.flatten(net_h53)

        feature = tf.concat([global_max11, global_max12, global_max13, global_max21, global_max22, global_max23
                                , global_max31, global_max32, global_max33], axis=1)

        net_h6 = tf.layers.dense(feature, 1, activation=tf.identity)
        logits = net_h6
        net_h6 = tf.nn.sigmoid(net_h6)
    return net_h6, logits, feature