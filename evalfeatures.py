# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Evaluates a TFGAN trained CIFAR model."""
import os
import tensorflow as tf

from Utils import get_init_vector
from cifar import networkssate as networks, data_provider_sattelite as data_provider, data_provider_sattelite
from cifar import dcgansatelite as dcgan
from cifar import util


flags = tf.flags
FLAGS = tf.flags.FLAGS
tfgan = tf.contrib.gan

flags.DEFINE_string('master', '', 'Name of the TensorFlow master to use.')

flags.DEFINE_string('eval_dir', '/data/satellitegpu/result',
                    'Directory where the results are saved to.')

flags.DEFINE_string('dataset_dir', "/data/satellitegpu/", 'Location of data.')

flags.DEFINE_integer('num_images_generated', 210,
                     'Number of images to generate at once.')

flags.DEFINE_integer('num_inception_images', 10,
                     'The number of images to run through Inception at once.')

flags.DEFINE_boolean('eval_real_images', False,
                     'If `True`, run Inception network on real images.')

flags.DEFINE_boolean('conditional_eval', True,
                     'If `True`, set up a conditional GAN.')

flags.DEFINE_boolean('eval_frechet_inception_distance', True,
                     'If `True`, compute Frechet Inception distance using real '
                     'images and generated images.')

flags.DEFINE_integer('num_images_per_class', 10,
                     'When a conditional generator is used, this is the number '
                     'of images to display per class.')

flags.DEFINE_integer('max_number_of_evaluations', None,
                     'Number of times to run evaluation. If `None`, run '
                     'forever.')

flags.DEFINE_integer('batch_size', 64, 'The number of images in each batch.')
flags.DEFINE_boolean('write_to_disk', True, 'If `True`, run images to disk.')
flags.DEFINE_integer('generator_init_vector_size', 100, 'Generator initialization vector size')
flags.DEFINE_integer("output_size", 256, "The size of the output images to produce [64]")
flags.DEFINE_integer("c_dim", 3, "Dimension of image color. [3]")
flags.DEFINE_string('checkpoint_dir', '/data/satellitegpu/train_log4',
                    'Directory where the model was written to.')


INPUT_TENSOR = 'g/in:0'
OUTPUT_TENSOR = 'logits:0'
MODEL_GRAPH_DEF = "/data/satellitegpu/train_log4/graph.pbtxt"

def geoxt_score(images, graph_def_filename=None, input_tensor=INPUT_TENSOR,
                output_tensor=OUTPUT_TENSOR, num_batches=1):

  images.shape.assert_is_compatible_with([None, 256, 256, 3])

  graph_def = _graph_def_from_par_or_disk(graph_def_filename)
  mnist_classifier_fn = lambda x: tfgan.eval.run_image_classifier(  # pylint: disable=g-long-lambda
      x, graph_def, input_tensor, output_tensor)

  score = tfgan.eval.classifier_score(
      images, mnist_classifier_fn, num_batches)
  score.shape.assert_is_compatible_with([])

  return score

def _graph_def_from_par_or_disk(filename):
  if filename is None:
    return tfgan.eval.get_graph_def_from_resource(MODEL_GRAPH_DEF)
  else:
    return tfgan.eval.get_graph_def_from_disk(MODEL_GRAPH_DEF)

def main(_, run_eval_loop=True):
  # Fetch and generate images to run through Inception.
  with tf.name_scope('inputs'):
      real_images, one_hot_labels, _, num_classes = data_provider_sattelite.provide_data(
        FLAGS.batch_size, FLAGS.dataset_dir)



def _get_real_data(num_images_generated, dataset_dir):
  """Get real images."""
  data, _, _, num_classes = data_provider.provide_data(
      num_images_generated, dataset_dir)
  return data, num_classes


def _get_generated_data(num_images_generated, conditional_eval, num_classes):
  """Get generated images."""
  noise = get_init_vector(FLAGS.generator_init_vector_size, FLAGS.batch_size)
  # If conditional, generate class-specific images.
  if conditional_eval:
    conditioning = util.get_generator_conditioning(
        num_images_generated, num_classes)
    generator_inputs = (noise, conditioning)
    generator_fn = networks.conditional_generator
  else:
    generator_inputs = noise
    generator_fn = networks.generator

  with tf.variable_scope('Generator'):
    data = generator_fn(generator_inputs, is_training=False)

  return data


if __name__ == '__main__':
  tf.app.run()
