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
from tensorflow.contrib.learn.python.learn.utils.inspect_checkpoint import print_tensors_in_checkpoint_file

from Utils import get_init_vector
from cifar import networkssate as networks, data_provider_sattelite as data_provider, data_provider_sattelite
from cifar import dcgansatelite as dcgan
from cifar import util
from tensorflow.contrib import predictor, slim
import matplotlib.pyplot as plt
import numpy as np

flags = tf.flags
FLAGS = tf.flags.FLAGS
tfgan = tf.contrib.gan

flags.DEFINE_string('master', '', 'Name of the TensorFlow master to use.')

flags.DEFINE_string('eval_dir', '/data/satellitegpu/result1',
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
flags.DEFINE_string('checkpoint_dir', '/data/satellitegpu/train_log12',
                    'Directory where the model was written to.')

#def getClassIndex(classList, ):



def main(_, run_eval_loop=True):
  tf.reset_default_graph()
  dataset_type = "test"
  classesList = ["agricultural", 'airplane', 'baseballdiamond', 'beach', 'buildings', 'chaparral'
      , 'denseresidential', 'forest', 'freeway', 'golfcourse', 'harbor', 'intersection', 'mediumresidential',
                 'mobilehomepark', 'overpass', 'parkinglot', 'river', 'runway', 'sparseresidential',
                 'storagetanks', 'tenniscourt']

  def name_in_checkpoint(var):
      if "Discriminator/" in var.op.name:
          return var.op.name.replace("Discriminator/", "Discriminator/Discriminator/")


  with tf.name_scope('inputs1'):

      real_images, one_hot_labels, _, num_classes = data_provider_sattelite.provide_data(
          FLAGS.batch_size, FLAGS.dataset_dir, split_name=dataset_type)


      logits, end_points_des, feature, net_h7 = dcgan.discriminator(real_images)

      variables_to_restore = slim.get_model_variables()
      variables_to_restore = {name_in_checkpoint(var): var for var in variables_to_restore}
      restorer = tf.train.Saver(variables_to_restore)

      #variables_to_restore = slim.get_model_variables()
      #restorer = tf.train.Saver(variables_to_restore)

      # Calculate predictions.
      #init_op = tf.global_variables_initializer()
  with tf.Session() as sess:
          sess.run(tf.global_variables_initializer())
          sess.run(tf.local_variables_initializer())

      #tf.get_variable_scope().reuse_variables()


          #print (sess.run(feature, feed_dict={real_images:real_images}))
          #img, lbl = sess.run([real_images, one_hot_labels])
          #print (sess.run(lbl))
          coord = tf.train.Coordinator()
          threads = tf.train.start_queue_runners(coord=coord)
          for batch_index in range(1):
              img, lbl = sess.run([real_images, one_hot_labels])
              for image in img:
                plt.imshow(image)
                plt.show()
                print(img.shape)
          # Stop the threads
          coord.request_stop()

          # Wait for threads to stop
          coord.join(threads)
          sess.close()



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
