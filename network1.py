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
    layer = tf.layers.batch_normalization(layer, epsilon=1e-12, gamma_initializer=gamma_init,
                                          training=is_training)
    return tf.nn.leaky_relu(layer, 0.2)
    #return layer

def generator_simplified_api(inputs, batch_size, is_train=True, reuse=False):
    image_size = 64
    k = 4
    # 128, 64, 32, 16
    s2, s4, s8, s16, s32, s64 = int(image_size/2), int(image_size/4), int(image_size/8),\
                                int(image_size/16), int(image_size/32), int(image_size/64)


    gf_dim = 16 # Dimension of gen filters in first conv layer. [64]

    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)

    with tf.variable_scope("generator", reuse=reuse):

        depth_of_h0 = 256
        net_h01 = tf.layers.dense( inputs, depth_of_h0, activation=tf.identity)
        net_h01 = tf.reshape(net_h01, shape=(-1, 1, 1, depth_of_h0))
        net_h01 = batch_normalization_layer(net_h01, gamma_init, 'g/h0/batch_norm', is_train, is_train, reuse=reuse)

        net_h02 = tf.layers.dense(inputs, depth_of_h0, activation=tf.identity)
        net_h02 = tf.reshape(net_h02, shape=(-1, 1, 1, depth_of_h0))
        net_h02 = batch_normalization_layer(net_h02, gamma_init, 'g/h0/batch_norm', is_train, is_train, reuse=reuse)

        net_h03 = tf.layers.dense(inputs, depth_of_h0, activation=tf.identity)
        net_h03 = tf.reshape(net_h03, shape=(-1, 1, 1, depth_of_h0))
        net_h03 = batch_normalization_layer(net_h03, gamma_init, 'g/h0/batch_norm', is_train, is_train, reuse=reuse)

        depth_of_h1 = int(depth_of_h0/2)
        net_h11 = tf.layers.conv2d_transpose(net_h01, depth_of_h1, [1, 1], strides=(2,2), padding='SAME', activation=None)
        net_h11 = batch_normalization_layer(net_h11, gamma_init, 'g/h1/batch_norm', is_train, is_train, reuse=reuse)
        net_h12 = tf.layers.conv2d_transpose(net_h02, depth_of_h1, [3, 3], strides=(2, 2), padding='SAME',
                                             activation=None)
        net_h12 = batch_normalization_layer(net_h12, gamma_init, 'g/h1/batch_norm', is_train, is_train, reuse=reuse)

        net_h13 = tf.layers.conv2d_transpose(net_h03, depth_of_h1, [5, 5], strides=(2, 2), padding='SAME',
                                             activation=None)
        net_h13 = batch_normalization_layer(net_h13, gamma_init, 'g/h1/batch_norm', is_train, is_train, reuse=reuse)

        depth_of_h2 = int(depth_of_h1 / 2)
        net_h21 = tf.layers.conv2d_transpose(net_h11, depth_of_h2, [1, 1], strides=(2,2), padding='SAME', activation=None)
        net_h21 = batch_normalization_layer(net_h21, gamma_init, 'g/h2/batch_norm', is_train, is_train, reuse=reuse)
        net_h22 = tf.layers.conv2d_transpose(net_h12, depth_of_h2, [3, 3], strides=(2, 2), padding='SAME',
                                             activation=None)
        net_h22 = batch_normalization_layer(net_h22, gamma_init, 'g/h2/batch_norm', is_train, is_train, reuse=reuse)
        net_h23 = tf.layers.conv2d_transpose(net_h13, depth_of_h2, [5, 5], strides=(2, 2), padding='SAME',
                                             activation=None)
        net_h23 = batch_normalization_layer(net_h23, gamma_init, 'g/h2/batch_norm', is_train, is_train, reuse=reuse)

        depth_of_h3 = int(depth_of_h2 / 2)
        net_h31 = tf.layers.conv2d_transpose(net_h21, depth_of_h3, [1, 1], strides=(2,2), padding='SAME', activation=None)
        net_h31 = batch_normalization_layer(net_h31, gamma_init, 'g/h3/batch_norm', is_train, is_train, reuse=reuse)
        net_h32 = tf.layers.conv2d_transpose(net_h22, depth_of_h3, [3, 3], strides=(2, 2), padding='SAME',
                                             activation=None)
        net_h32 = batch_normalization_layer(net_h32, gamma_init, 'g/h3/batch_norm', is_train, is_train, reuse=reuse)
        net_h33 = tf.layers.conv2d_transpose(net_h23, depth_of_h3, [5, 5], strides=(2, 2), padding='SAME',
                                             activation=None)
        net_h33 = batch_normalization_layer(net_h33, gamma_init, 'g/h3/batch_norm', is_train, is_train, reuse=reuse)

        depth_of_h4 = int(depth_of_h3 / 2)
        net_h41 = tf.layers.conv2d_transpose(net_h31, depth_of_h4, [1, 1], strides=(2,2), padding='SAME', activation=None)
        net_h41 = batch_normalization_layer(net_h41, gamma_init, 'g/h4/batch_norm', is_train, is_train, reuse=reuse)
        net_h42 = tf.layers.conv2d_transpose(net_h32, depth_of_h4, [3,3], strides=(2, 2), padding='SAME',
                                             activation=None)
        net_h42 = batch_normalization_layer(net_h42, gamma_init, 'g/h4/batch_norm', is_train, is_train, reuse=reuse)
        net_h43 = tf.layers.conv2d_transpose(net_h33, depth_of_h4, [5,5], strides=(2, 2), padding='SAME',
                                             activation=None)
        net_h43 = batch_normalization_layer(net_h43, gamma_init, 'g/h4/batch_norm', is_train, is_train, reuse=reuse)

        depth_of_h5 = int(depth_of_h4/2)
        net_h51 = tf.layers.conv2d_transpose(net_h41, depth_of_h5, [1,1], strides=(2, 2), padding='SAME', activation=None)
        net_h51 = batch_normalization_layer(net_h51, gamma_init, 'g/h5/batch_norm', is_train, is_train, reuse=reuse)
        net_h52 = tf.layers.conv2d_transpose(net_h42, depth_of_h5, [3, 3], strides=(2, 2), padding='SAME',
                                             activation=None)
        net_h52 = batch_normalization_layer(net_h52, gamma_init, 'g/h5/batch_norm', is_train, is_train, reuse=reuse)
        net_h53 = tf.layers.conv2d_transpose(net_h43, depth_of_h5, [5, 5], strides=(2, 2), padding='SAME',
                                             activation=None)
        net_h53 = batch_normalization_layer(net_h53, gamma_init, 'g/h5/batch_norm', is_train, is_train, reuse=reuse)

        #net_h60 = tf.concat(axis=3, values=[net_h51, net_h52, net_h53])
        net_h61 = tf.layers.conv2d_transpose(net_h51, 3, [1,1], strides=(2, 2), padding='SAME',activation=None)
        net_h62 = tf.layers.conv2d_transpose(net_h52, 3, [3,3], strides=(2, 2), padding='SAME',activation=None)
        net_h63 = tf.layers.conv2d_transpose(net_h53, 3, [5,5], strides=(2, 2), padding='SAME',activation=None)

        logits = net_h63
        net_h6 = tf.nn.tanh(net_h63)
    return net_h6, logits

def conditional_discriminator_simplified_api(inputs, condition, is_train=True, reuse=False):
    conditinal_input = tf.concat([inputs, condition], 0)
    return discriminator_simplified_api(conditinal_input, is_train, reuse)

def discriminator_simplified_api(inputs, is_train=True, reuse=False):
    k = 3
    df_dim = 16 # Dimension of discrim filters in first conv layer. [64]
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)

    with tf.variable_scope("discriminator", reuse=reuse):

        net_h01 = tf.layers.conv2d(inputs, df_dim, [1, 1], strides=(2,2), padding='SAME')
        net_h02 = tf.layers.conv2d(inputs, df_dim, [3, 3], strides=(2,2), padding='SAME')
        net_h03 = tf.layers.conv2d(inputs, df_dim, [5, 5], strides=(2,2), padding='SAME')

        #net_h0 = tf.concat(axis=3, values=[net_h01, net_h02, net_h03])

        depth_of_h1 = df_dim*2
        net_h11 = tf.layers.conv2d(net_h01, depth_of_h1, [1, 1], strides=(2,2), padding='SAME',
                                          activation=None)
        net_h11 = batch_normalization_layer(net_h11, gamma_init, 'd/h1/batch_norm'
                                           , is_train, is_train, reuse=reuse)
        net_h12 = tf.layers.conv2d(net_h02, depth_of_h1, [3, 3], strides=(2, 2), padding='SAME',
                                  activation=None)
        net_h12 = batch_normalization_layer(net_h12, gamma_init, 'd/h1/batch_norm'
                                           , is_train, is_train, reuse=reuse)

        net_h13 = tf.layers.conv2d(net_h03, depth_of_h1, [5, 5], strides=(2, 2), padding='SAME',
                                  activation=None)
        net_h13 = batch_normalization_layer(net_h13, gamma_init, 'd/h1/batch_norm'
                                           , is_train, is_train, reuse=reuse)

        #net_h1 = tf.concat(axis=3, values=[net_h11, net_h12, net_h13])

        depth_of_h2 = depth_of_h1*2
        net_h21 = tf.layers.conv2d(net_h11, depth_of_h2, [1, 1], strides=(2,2), padding='SAME',
                                          activation=None)
        net_h21 = batch_normalization_layer(net_h21, gamma_init, 'd/h2/batch_norm'
                                           , is_train, is_train, reuse=reuse)
        net_h22 = tf.layers.conv2d(net_h12, depth_of_h2, [3, 3], strides=(2, 2), padding='SAME',
                                   activation=None)
        net_h22 = batch_normalization_layer(net_h22, gamma_init, 'd/h2/batch_norm'
                                            , is_train, is_train, reuse=reuse)
        net_h23 = tf.layers.conv2d(net_h13, depth_of_h2, [5, 5], strides=(2, 2), padding='SAME',
                                   activation=None)
        net_h23 = batch_normalization_layer(net_h23, gamma_init, 'd/h2/batch_norm'
                                            , is_train, is_train, reuse=reuse)

        depth_of_h3 = depth_of_h2*2
        net_h31 = tf.layers.conv2d(net_h21, depth_of_h3, [1, 1], strides=(2,2), padding='SAME',
                                          activation=None)
        net_h31 = batch_normalization_layer(net_h31, gamma_init, 'd/h3/batch_norm'
                                           , is_train, is_train, reuse=reuse)
        net_h32 = tf.layers.conv2d(net_h22, depth_of_h3, [2, 2], strides=(2, 2), padding='SAME',
                                   activation=None)
        net_h32 = batch_normalization_layer(net_h32, gamma_init, 'd/h3/batch_norm'
                                            , is_train, is_train, reuse=reuse)
        net_h33 = tf.layers.conv2d(net_h23, depth_of_h3, [3, 3], strides=(2, 2), padding='SAME',
                                   activation=None)
        net_h33 = batch_normalization_layer(net_h33, gamma_init, 'd/h3/batch_norm'
                                            , is_train, is_train, reuse=reuse)

        #global_max1 = tf.layers.max_pooling2d( net_h3, [4,4], strides=1, padding='SAME')
        #global_max1 = tf.layers.flatten(net_h31)
        depth_of_h4 = depth_of_h3 * 2
        net_h41 = tf.layers.conv2d(net_h31, depth_of_h4, [1, 1], strides=(2,2), padding='SAME',
                                          activation=None)
        net_h41 = batch_normalization_layer(net_h41, gamma_init, 'd/h4/batch_norm'
                                           , is_train, is_train, reuse=reuse)
        net_h42 = tf.layers.conv2d(net_h32, depth_of_h4, [3, 3], strides=(2, 2), padding='SAME',
                                   activation=None)
        net_h42 = batch_normalization_layer(net_h42, gamma_init, 'd/h4/batch_norm'
                                            , is_train, is_train, reuse=reuse)
        net_h43 = tf.layers.conv2d(net_h33, depth_of_h4, [5, 5], strides=(2, 2), padding='SAME',
                                   activation=None)
        net_h43 = batch_normalization_layer(net_h43, gamma_init, 'd/h4/batch_norm'
                                            , is_train, is_train, reuse=reuse)

        #global_max2 = tf.layers.max_pooling2d(net_h4, [2, 2], strides=1, padding='SAME')
        global_max21 = tf.layers.flatten(net_h41, name='d/h4/flatten')
        global_max22 = tf.layers.flatten(net_h42, name='d/h4/flatten')
        global_max23 = tf.layers.flatten(net_h43, name='d/h4/flatten')

        depth_of_h5 = depth_of_h4 * 2
        net_h51 = tf.layers.conv2d(net_h41, depth_of_h5, [1, 1], strides=(2,2), padding='SAME',
                                          activation=None)
        net_h51 = batch_normalization_layer(net_h51, gamma_init, 'd/h5/batch_norm'
                                           , is_train, is_train, reuse=reuse)
        net_h52 = tf.layers.conv2d(net_h42, depth_of_h5, [3, 3], strides=(2, 2), padding='SAME',
                                   activation=None)
        net_h52 = batch_normalization_layer(net_h52, gamma_init, 'd/h5/batch_norm'
                                            , is_train, is_train, reuse=reuse)
        net_h53 = tf.layers.conv2d(net_h43, depth_of_h5, [5, 5], strides=(2, 2), padding='SAME',
                                   activation=None)
        net_h53 = batch_normalization_layer(net_h53, gamma_init, 'd/h5/batch_norm'
                                            , is_train, is_train, reuse=reuse)

        global_max31 = tf.layers.flatten(net_h51)
        global_max32 = tf.layers.flatten(net_h52)
        global_max33 = tf.layers.flatten(net_h53)
        feature = tf.concat([global_max21, global_max22, global_max23, global_max31,
                             global_max32, global_max33], axis=1)

        net_h6 = tf.layers.dense(feature, 1, activation=tf.identity)
        logits = net_h6
        net_h6 = tf.nn.sigmoid(net_h6)
    return net_h6, logits, feature