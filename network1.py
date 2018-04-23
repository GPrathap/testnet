#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 20:52:00 2017

@author: ldy
"""


import numpy as np

import tensorflow as tf


from utils import *


def conditional_generator_simplified_api(inputs, condition, batch_size, is_train=True, reuse=False):
    conditinal_input = tf.concat([inputs, condition], 0)
    return generator_simplified_api(conditinal_input, batch_size, is_train, reuse)


def batch_normalization_layer(layer, gamma_init, scope, is_training, is_trainable, reuse):
    layer = tf.layers.batch_normalization(layer, epsilon=1e-12, gamma_initializer=gamma_init, name=scope,
                                          training=is_training, trainable=is_trainable, reuse=reuse)
    return tf.nn.leaky_relu(layer, 0.2)
    #return layer

def generator_simplified_api(inputs, batch_size, is_train=True, reuse=False):
    image_size = 256
    k = 4
    # 128, 64, 32, 16
    s2, s4, s8, s16, s32, s64 = int(image_size/2), int(image_size/4), int(image_size/8),\
                                int(image_size/16), int(image_size/32), int(image_size/64)


    gf_dim = 16 # Dimension of gen filters in first conv layer. [64]

    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)

    with tf.variable_scope("generator", reuse=reuse):

        net_h0 = tf.layers.dense( inputs, gf_dim * 32 * s64 * s64,
            activation=tf.identity, reuse=reuse, trainable=is_train, name='g/h0/dense')
        net_h0 = tf.reshape(net_h0, shape=(-1, s64, s64, gf_dim*32))
        net_h0 = batch_normalization_layer(net_h0, gamma_init, 'g/h0/batch_norm'
                                           , is_train, is_train, reuse=reuse)

        net_h1 = tf.layers.conv2d_transpose(
            net_h0, gf_dim*16, [k, k], strides=(2,2), padding='SAME', activation=None,
            trainable=is_train, name='g/h1/conv2d_transpose')
        net_h1 = batch_normalization_layer(net_h1, gamma_init, 'g/h1/batch_norm'
                                           , is_train, is_train, reuse=reuse)

        net_h2 = tf.layers.conv2d_transpose(
            net_h1, gf_dim * 8, [k, k], strides=(2,2), padding='SAME', activation=None,
            trainable=is_train, name='g/h2/conv2d_transpose')
        net_h2 = batch_normalization_layer(net_h2, gamma_init, 'g/h2/batch_norm'
                                           , is_train, is_train, reuse=reuse)

        net_h3 = tf.layers.conv2d_transpose(
            net_h2, gf_dim * 4, [k, k], strides=(2,2), padding='SAME', activation=None,
            trainable=is_train, name='g/h3/conv2d_transpose')
        net_h3 = batch_normalization_layer(net_h3, gamma_init, 'g/h3/batch_norm'
                                           , is_train, is_train, reuse=reuse)

        net_h4 = tf.layers.conv2d_transpose(
            net_h3, gf_dim * 2, [k, k], strides=(2,2), padding='SAME', activation=None,
            trainable=is_train, name='g/h4/conv2d_transpose')
        net_h4 = batch_normalization_layer(net_h4, gamma_init, 'g/h4/batch_norm'
                                           , is_train, is_train, reuse=reuse)

        net_h5 = tf.layers.conv2d_transpose(
            net_h4, gf_dim * 1, [k, k], strides=(2,2), padding='SAME', activation=None,
            trainable=is_train, name='g/h5/conv2d_transpose')
        net_h5 = batch_normalization_layer(net_h5, gamma_init, 'g/h5/batch_norm'
                                           , is_train, is_train, reuse=reuse)

        net_h6 = tf.layers.conv2d_transpose(
            net_h5, 3, [k, k], strides=(2,2), padding='SAME', activation=None,
            trainable=is_train, name='g/h6/conv2d_transpose')

        logits = net_h6
        net_h6 = tf.nn.tanh(net_h6)
    return net_h6, logits

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def conditional_discriminator_simplified_api(inputs, condition, is_train=True, reuse=False):
    conditinal_input = tf.concat([inputs, condition], 0)
    return discriminator_simplified_api(conditinal_input, is_train, reuse)

def discriminator_simplified_api(inputs, is_train=True, reuse=False):
    k = 5
    df_dim = 16 # Dimension of discrim filters in first conv layer. [64]
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)

    with tf.variable_scope("discriminator", reuse=reuse):

        net_h1 = tf.layers.conv2d(inputs, df_dim, [k, k], strides=(2,2), padding='SAME',
                                           trainable=is_train,
                                          activation= lambda x: tf.nn.leaky_relu(x, 0.2), name='g/h0/conv2d')

        net_h1 = tf.layers.conv2d(net_h1, df_dim*2, [k, k], strides=(2,2), padding='SAME',
                                            trainable=is_train,
                                          activation=None, name='d/h1/conv2d')
        net_h1 = batch_normalization_layer(net_h1, gamma_init, 'd/h1/batch_norm'
                                           , is_train, is_train, reuse=reuse)

        net_h2 = tf.layers.conv2d(net_h1, df_dim * 4, [k, k], strides=(2,2), padding='SAME',
                                            trainable=is_train,
                                          activation=None, name='d/h2/conv2d')
        net_h2 = batch_normalization_layer(net_h2, gamma_init, 'd/h2/batch_norm'
                                           , is_train, is_train, reuse=reuse)

        net_h3 = tf.layers.conv2d(net_h2, df_dim * 8, [k, k], strides=(2,2), padding='SAME',
                                           trainable=is_train,
                                          activation=None, name='d/h3/conv2d')
        net_h3 = batch_normalization_layer(net_h3, gamma_init, 'd/h3/batch_norm'
                                           , is_train, is_train, reuse=reuse)

        global_max1 = tf.layers.max_pooling2d( net_h3, [4,4], strides=1, padding='SAME', name='d/h3/max_pool2d')
        global_max1 = tf.layers.flatten(global_max1, name='d/h3/flatten')

        net_h4 = tf.layers.conv2d(net_h3, df_dim * 16, [k, k], strides=(2,2), padding='SAME',
                                           trainable=is_train,
                                          activation=None, name='d/h4/conv2d')
        net_h4 = batch_normalization_layer(net_h4, gamma_init, 'd/h4/batch_norm'
                                           , is_train, is_train, reuse=reuse)

        global_max2 = tf.layers.max_pooling2d(net_h4, [2, 2], strides=1, padding='SAME', name='d/h4/max_pool2d')
        global_max2 = tf.layers.flatten(global_max2, name='d/h4/flatten')

        net_h5 = tf.layers.conv2d(net_h4, df_dim * 32, [k, k], strides=(2,2), padding='SAME',
                                            trainable=is_train,
                                          activation=None, name='d/h5/conv2d')
        net_h5 = batch_normalization_layer(net_h5, gamma_init, 'd/h5/batch_norm'
                                           , is_train, is_train, reuse=reuse)

        global_max3 = tf.layers.flatten(net_h5, name='d/h5/flatten')
        feature = tf.concat([global_max1, global_max2, global_max3], axis=1, name='d/h5/concat')

        net_h6 = tf.layers.dense(feature, 1, activation=tf.identity, reuse=reuse,
                                                   trainable=is_train, name='d/h6/dense')
        logits = net_h6
        net_h6 = tf.nn.sigmoid(net_h6)
    return net_h6, logits, feature