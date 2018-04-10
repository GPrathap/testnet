# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""DCGAN generator and discriminator from https://arxiv.org/abs/1511.06434."""

from math import log

from six.moves import xrange

import tensorflow as tf
from six.moves import xrange
from tensorlayer.layers import InputLayer, Conv2d, BatchNormLayer, MaxPool2d, FlattenLayer, ConcatLayer, DenseLayer, \
    ReshapeLayer, DeConv2d

import tensorlayer as tl

slim = tf.contrib.slim
layers = tf.layers


def _validate_image_inputs(inputs):
  inputs.get_shape().assert_has_rank(4)
  inputs.get_shape()[1:3].assert_is_fully_defined()
  if inputs.get_shape()[1] != inputs.get_shape()[2]:
    raise ValueError('Input tensor does not have equal width and height: ',
                     inputs.get_shape()[1:3])
  width = inputs.get_shape().as_list()[1]
  if log(width, 2) != int(log(width, 2)):
    raise ValueError('Input tensor `width` is not a power of 2: ', width)


# TODO(joelshor): Use fused batch norm by default. Investigate why some GAN
# setups need the gradient of gradient FusedBatchNormGrad.
def discriminator(inputs,
                  depth=8,
                  is_training=True,
                  reuse=None,
                  scope='Discriminator',
                  fused_batch_norm=False):
  """Discriminator network for DCGAN.

  Construct discriminator network from inputs to the final endpoint.

  Args:
    inputs: A tensor of size [batch_size, height, width, channels]. Must be
      floating point.
    depth: Number of channels in first convolution layer.
    is_training: Whether the network is for training or not.
    reuse: Whether or not the network variables should be reused. `scope`
      must be given to be reused.
    scope: Optional variable_scope.
    fused_batch_norm: If `True`, use a faster, fused implementation of
      batch norm.

  Returns:
    logits: The pre-softmax activations, a tensor of size [batch_size, 1]
    end_points: a dictionary from components of the network to their activation.

  Raises:
    ValueError: If the input image shape is not 4-dimensional, if the spatial
      dimensions aren't defined at graph construction time, if the spatial
      dimensions aren't square, or if the spatial dimensions aren't a power of
      two.
  """


  normalizer_fn = slim.batch_norm
  normalizer_fn_args = {
      'is_training': is_training,
      'zero_debias_moving_mean': True,
      'fused': fused_batch_norm,
  }

  _validate_image_inputs(inputs)
  inp_shape = inputs.get_shape().as_list()[1]

  k = 5
  df_dim = 16  # Dimension of discrim filters in first conv layer. [64]
  w_init = tf.random_normal_initializer(stddev=0.02)
  gamma_init = tf.random_normal_initializer(1., 0.02)

  end_points = {}
  with tf.variable_scope(scope, values=[inputs], reuse=reuse) as scope:
    with slim.arg_scope([normalizer_fn], **normalizer_fn_args):
      with slim.arg_scope([slim.conv2d],
                          stride=2,
                          kernel_size=4,
                          activation_fn=tf.nn.leaky_relu):
        net = inputs

        scope = 'conv0'
        net_h0 = slim.convolution2d(net, df_dim, normalizer_fn=normalizer_fn, scope=scope)
        end_points[scope] = net_h0

        scope = 'conv1'
        net_h1 = slim.convolution2d(net_h0, 2*df_dim, normalizer_fn=normalizer_fn, scope=scope)
        end_points[scope] = net_h1

        scope = 'conv2'
        net_h2 = slim.convolution2d(net_h1, 4 * df_dim, normalizer_fn=normalizer_fn, scope=scope)
        end_points[scope] = net_h2

        scope = 'conv3'
        net_h3 = slim.convolution2d(net_h2, 8 * df_dim, normalizer_fn=normalizer_fn, scope=scope)
        end_points[scope] = net_h3

        global_max1 = slim.max_pool2d(net_h3, kernel_size=(4, 4), stride=1, padding='SAME', scope='maxpool1')
        global_max1 = slim.flatten(global_max1, scope='flatten1')

        scope = 'conv4'
        net_h4 = slim.convolution2d(net_h3, 16 * df_dim, normalizer_fn=normalizer_fn, scope=scope)
        end_points[scope] = net_h4

        global_max2 = slim.max_pool2d(net_h4, kernel_size=(2, 2), stride=1, padding='SAME', scope='maxpool2')
        global_max2 = slim.flatten(global_max2, scope='flatten2')

        scope = 'conv5'
        net_h5 = slim.convolution2d(net_h4, 16 * df_dim, normalizer_fn=normalizer_fn, scope=scope)
        end_points[scope] = net_h5

        global_max3 =slim.flatten(net_h5, scope='flatten3')

        feature = tf.concat(values = [global_max1, global_max2, global_max3], axis=1, name='d/concat_layer1')
        scope="conv6"
        net_h6 = slim.fully_connected(feature, num_outputs=int(feature.shape[1]//64), activation_fn=tf.identity,
                    normalizer_fn=None, scope='fully_connected_layer1')
        end_points[scope] = net_h6

        scope = "conv7"
        net_h7 = slim.fully_connected(net_h6, num_outputs=1, activation_fn=tf.identity,
                                      normalizer_fn=None, scope='fully_connected_layer2')
        end_points[scope] = net_h7

        logits = net_h7
        net_h7.outputs = tf.nn.sigmoid(net_h7)
        end_points['logits'] = logits

        return logits, end_points, feature, net_h7


# TODO(joelshor): Use fused batch norm by default. Investigate why some GAN
# setups need the gradient of gradient FusedBatchNormGrad.
def generator(inputs,
              depth=8,
              final_size=256,
              num_outputs=3,
              is_training=True,
              reuse=None,
              scope='Generator',
              fused_batch_norm=False):
  """Generator network for DCGAN.

  Construct generator network from inputs to the final endpoint.

  Args:
    inputs: A tensor with any size N. [batch_size, N]
    depth: Number of channels in last deconvolution layer.
    final_size: The shape of the final output.
    num_outputs: Number of output features. For images, this is the number of
      channels.
    is_training: whether is training or not.
    reuse: Whether or not the network has its variables should be reused. scope
      must be given to be reused.
    scope: Optional variable_scope.
    fused_batch_norm: If `True`, use a faster, fused implementation of
      batch norm.

  Returns:
    logits: the pre-softmax activations, a tensor of size
      [batch_size, 32, 32, channels]
    end_points: a dictionary from components of the network to their activation.

  Raises:
    ValueError: If `inputs` is not 2-dimensional.
    ValueError: If `final_size` isn't a power of 2 or is less than 8.
  """
  normalizer_fn = slim.batch_norm
  normalizer_fn_args = {
      'is_training': is_training,
      'zero_debias_moving_mean': True,
      'fused': fused_batch_norm,
  }
  end_points = {}
  batch_size = 64
  inputs.get_shape().assert_has_rank(2)
  if log(final_size, 2) != int(log(final_size, 2)):
    raise ValueError('`final_size` (%i) must be a power of 2.' % final_size)
  if final_size < 8:
    raise ValueError('`final_size` (%i) must be greater than 8.' % final_size)

  image_size = 256
  k = 4
  # 128, 64, 32, 16
  s2, s4, s8, s16, s32, s64 = int(image_size / 2), int(image_size / 4), int(image_size / 8), int(image_size / 16), int(
      image_size / 32), int(image_size / 64)

  gf_dim = 16  # Dimension of gen filters in first conv layer. [64]

  w_init = tf.random_normal_initializer(stddev=0.02)
  gamma_init = tf.random_normal_initializer(1., 0.02)

  with tf.variable_scope(scope, reuse=reuse):
      tl.layers.set_name_reuse(reuse)

      net_in = InputLayer(inputs, name='g/in')
      net_h0 = DenseLayer(net_in, n_units=gf_dim * 32 * s64 * s64, W_init=w_init,
                          act=tf.identity, name='g/h0/lin')
      net_h0 = ReshapeLayer(net_h0, shape=[-1, s64, s64, gf_dim * 32], name='g/h0/reshape')
      net_h0 = BatchNormLayer(net_h0, act=tf.nn.relu, is_train=is_training,
                              gamma_init=gamma_init, name='g/h0/batch_norm')
      scope = 'deconv0'
      end_points[scope] = net_h0

      net_h1 = DeConv2d(net_h0, gf_dim * 16, (k, k), out_size=(s32, s32), strides=(2, 2),
                        padding='SAME', batch_size=batch_size, act=None, W_init=w_init, name='g/h1/decon2d')
      net_h1 = BatchNormLayer(net_h1, act=tf.nn.relu, is_train=is_training,
                              gamma_init=gamma_init, name='g/h1/batch_norm')
      scope = 'deconv1'
      end_points[scope] = net_h1

      net_h2 = DeConv2d(net_h1, gf_dim * 8, (k, k), out_size=(s16, s16), strides=(2, 2),
                        padding='SAME', batch_size=batch_size, act=None, W_init=w_init, name='g/h2/decon2d')
      net_h2 = BatchNormLayer(net_h2, act=tf.nn.relu, is_train=is_training,
                              gamma_init=gamma_init, name='g/h2/batch_norm')
      scope = 'deconv2'
      end_points[scope] = net_h2

      net_h3 = DeConv2d(net_h2, gf_dim * 4, (k, k), out_size=(s8, s8), strides=(2, 2),
                        padding='SAME', batch_size=batch_size, act=None, W_init=w_init, name='g/h3/decon2d')
      net_h3 = BatchNormLayer(net_h3, act=tf.nn.relu, is_train=is_training,
                              gamma_init=gamma_init, name='g/h3/batch_norm')
      scope = 'deconv3'
      end_points[scope] = net_h3

      net_h4 = DeConv2d(net_h3, gf_dim * 2, (k, k), out_size=(s4, s4), strides=(2, 2),
                        padding='SAME', batch_size=batch_size, act=None, W_init=w_init, name='g/h4/decon2d')
      net_h4 = BatchNormLayer(net_h4, act=tf.nn.relu, is_train=is_training,
                              gamma_init=gamma_init, name='g/h4/batch_norm')
      scope = 'deconv4'
      end_points[scope] = net_h4

      net_h5 = DeConv2d(net_h4, gf_dim * 1, (k, k), out_size=(s2, s2), strides=(2, 2),
                        padding='SAME', batch_size=batch_size, act=None, W_init=w_init, name='g/h5/decon2d')
      net_h5 = BatchNormLayer(net_h5, act=tf.nn.relu, is_train=is_training,
                              gamma_init=gamma_init, name='g/h5/batch_norm')
      scope = 'deconv5'
      end_points[scope] = net_h5

      net_h6 = DeConv2d(net_h5, 3, (k, k), out_size=(image_size, image_size), strides=(2, 2),
                        padding='SAME', batch_size=batch_size, act=None, W_init=w_init, name='g/h6/decon2d')
      logits = net_h6.outputs
      net_h6.outputs = tf.nn.tanh(net_h6.outputs)
      scope = 'deconv6'
      end_points[scope] = net_h6

      scope = 'logits'
      end_points[scope] = logits

  return logits, end_points

  '''
  num_layers = int(log(final_size, 2)) - 1
  with tf.variable_scope(scope, values=[inputs], reuse=reuse) as scope:
    with slim.arg_scope([normalizer_fn], **normalizer_fn_args):
      with slim.arg_scope([slim.conv2d_transpose],
                          normalizer_fn=normalizer_fn,
                          stride=2,
                          kernel_size=4):
        net = tf.expand_dims(tf.expand_dims(inputs, 1), 1)

        # First upscaling is different because it takes the input vector.
        current_depth = depth * 2 ** (num_layers - 1)
        scope = 'deconv1'
        net = slim.conv2d_transpose(
            net, current_depth, stride=1, padding='VALID', scope=scope)
        end_points[scope] = net

        for i in xrange(2, num_layers):
          scope = 'deconv%i' % (i)
          current_depth = depth * 2 ** (num_layers - i)
          net = slim.conv2d_transpose(net, current_depth, scope=scope)
          end_points[scope] = net

        # Last layer has different normalizer and activation.
        scope = 'deconv%i' % (num_layers)
        net = slim.conv2d_transpose(
            net, depth, normalizer_fn=None, activation_fn=None, scope=scope)
        end_points[scope] = net

        # Convert to proper channels.
        scope = 'logits'
        logits = slim.conv2d(
            net,
            num_outputs,
            normalizer_fn=None,
            activation_fn=None,
            kernel_size=1,
            stride=1,
            padding='VALID',
            scope=scope)
        end_points[scope] = logits

        logits.get_shape().assert_has_rank(4)
        logits.get_shape().assert_is_compatible_with(
            [None, final_size, final_size, num_outputs])

        return logits, end_points

  '''


