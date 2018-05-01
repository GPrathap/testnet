import tensorflow as tf

from cifar.dataset_factory import get_dataset

slim = tf.contrib.slim


def provide_data(batch_size, dataset_dir, dataset_name='satellite',
                 split_name='train', one_hot=True):
  dataset = get_dataset(dataset_name, split_name, dataset_dir=dataset_dir)
  provider = slim.dataset_data_provider.DatasetDataProvider(
      dataset,
      common_queue_capacity=5 * batch_size,
      common_queue_min=batch_size,
      shuffle=(split_name == split_name))
  [image, label] = provider.get(['image', 'label'])
  # Preprocess the images.
  image = (tf.to_float(image) - 128.0) / 128.0
  # Creates a QueueRunner for the pre-fetching operation.
  images, labels = tf.train.batch(
      [image, label],
      batch_size=batch_size,
      num_threads=1,
      capacity=5 * batch_size)

  labels = tf.reshape(labels, [-1])
  if one_hot:
    labels = tf.one_hot(labels, dataset.num_classes)
  return images, labels, dataset.num_samples, dataset.num_classes


def float_image_to_uint8(image):
  image = (image * 128.0) + 128.0
  return tf.cast(image, tf.uint8)
