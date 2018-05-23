from six.moves import cPickle
import os
import sys
import tarfile
import glob
from six.moves import cPickle as pickle
import numpy as np
import scipy.misc
import tensorflow as tf

from dataset import dataset_utils

slim = tf.contrib.slim

class DataConvertor():
    def __init__(self, classes_list, image_size, dataset_name, dataset_storage_location, channels=3):
        self.classes_list = classes_list
        self.number_of_classes = len(self.classes_list)
        self.image_size = image_size
        self.channels = channels
        self.dataset_name = dataset_name
        self.dataset_storage_location = dataset_storage_location
        self.dataset_description = {
            'image': 'A [ '+ str(self.image_size) +'] color images.',
            'label': 'A single integer between 0 and '+ str(self.number_of_classes),
        }

        if not tf.gfile.Exists(self.dataset_storage_location):
            tf.gfile.MakeDirs(self.dataset_storage_location)

    def _add_to_tfrecord(self, filename, tfrecord_writer, offset=0):
      with tf.gfile.Open(filename, 'rb') as f:
        if sys.version_info < (3,):
          data = cPickle.load(f)
        else:
          data = cPickle.load(f, encoding='bytes')
      images = data['images']
      images = np.array(images)
      num_images = images.shape[0]
      images = images.reshape((num_images, self.channels, self.image_size, self.image_size))
      labels = data['labels']
      with tf.Graph().as_default():
        image_placeholder = tf.placeholder(dtype=tf.uint8)
        encoded_image = tf.image.encode_png(image_placeholder)
        with tf.Session('') as sess:
          for j in range(num_images):
            sys.stdout.write('\r>> Reading file [%s] image %d/%d' % (
                filename, offset + j + 1, offset + num_images))
            sys.stdout.flush()
            image = np.squeeze(images[j]).transpose((1, 2, 0))
            label = labels[j]
            png_string = sess.run(encoded_image,
                                  feed_dict={image_placeholder: image})
            example = dataset_utils.image_to_tfexample(
                png_string, b'png', self.image_size, self.image_size, label)
            tfrecord_writer.write(example.SerializeToString())
      return offset + num_images

    def _add_to_tfrecord_new(self, filename, tfrecord_writer, offset=0):
      with tf.gfile.Open(filename, 'rb') as f:
        if sys.version_info < (3,):
          data = cPickle.load(f)
        else:
          data = cPickle.load(f, encoding='bytes')
      images = data['images']
      labels = data['labels']
      images = np.array(images)
      num_images = images.shape[0]
      for X, y in zip(images,labels):
          # Feature contains a map of string to feature proto objects
          feature = {}
          feature['X'] = tf.train.Feature(float_list=tf.train.FloatList(value=X.flatten()))
          feature['y'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[y]))

          # Construct the Example proto object
          example = tf.train.Example(features=tf.train.Features(feature=feature))

          # Serialize the example to a string
          serialized = example.SerializeToString()

          # write the serialized objec to the disk
          tfrecord_writer.write(serialized)

      #tfrecord_writer.close()

    def create_files(self, path):
      datalist = {}
      datalist["images"] = []
      datalist["labels"] = []
      for filename in glob.iglob(path, recursive=True):
        image = scipy.misc.imread(filename).astype(np.float)
        image = scipy.misc.imresize(image, [self.image_size, self.image_size])
        currentClass = ""
        for cla in self.classes_list:
          if cla in filename:
            currentClass = cla
            break
        if currentClass == "":
          print("No class label is found")
        else:
          datalist["images"].append(image)
          datalist["labels"].append(self.classes_list.index(currentClass))
      return datalist


    def convert_into_tfrecord(self, dataset_path, is_train, is_dump=True):

      if is_train:
        if is_dump:
            datalist = self.create_files(dataset_path)
            dataset_storage_location = os.path.join(self.dataset_storage_location, self.dataset_name
                                                    + "_train.pickle")
            with open(dataset_storage_location, 'wb') as f:
                pickle.dump(datalist, f, protocol=pickle.HIGHEST_PROTOCOL)
        self.run(is_train)
      else:
          if is_dump:
              datalist = self.create_files(dataset_path)
              dataset_storage_location = os.path.join(self.dataset_storage_location, self.dataset_name
                                                      + "_test.pickle")
              with open(dataset_storage_location, 'wb') as f:
                 pickle.dump(datalist, f, protocol=pickle.HIGHEST_PROTOCOL)
          self.run(is_train)

    def _get_output_filename(self, split_name):
      return '%s/%s_%s.tfrecord' % (self.dataset_storage_location, self.dataset_name, split_name)


    def run(self, train=True):
      if train==True:
        training_filename = self._get_output_filename('train')
        with tf.python_io.TFRecordWriter(training_filename) as tfrecord_writer:
          offset = 0
          filename = os.path.join(self.dataset_storage_location, self.dataset_name + "_train.pickle")
          #offset = self._add_to_tfrecord(filename, tfrecord_writer, offset)
          offset = self._add_to_tfrecord_new(filename, tfrecord_writer, offset)
          labels_to_class_names = dict(zip(range(len(self.classes_list)), self.classes_list))
          self.write_label_file(labels_to_class_names)
          print('\nFinished converting the '+self.dataset_name+' dataset!')
      else:
        testing_filename = self._get_output_filename('test')
        with tf.python_io.TFRecordWriter(testing_filename) as tfrecord_writer:
          filename = os.path.join(self.dataset_storage_location, self.dataset_name + "_test.pickle")
          self._add_to_tfrecord_new(filename, tfrecord_writer)
        labels_to_class_names = dict(zip(range(len(self.classes_list)), self.classes_list))
        self.write_label_file(labels_to_class_names)
        print('\nFinished converting the  dataset!')

    def write_label_file(self, labels_to_class_names):
        labels_filename = os.path.join(self.dataset_storage_location, self.dataset_name + "_labels.txt")
        with tf.gfile.Open(labels_filename, 'w') as f:
            for label in labels_to_class_names:
                class_name = labels_to_class_names[label]
                f.write('%d:%s\n' % (label, class_name))


    def provide_data(self, batch_size, number_of_channels, split_name='train', one_hot=True):

        # Creates a dataset that reads all of the examples from filenames.
        filenames = [ self.get_tfrecord_file_location(self.dataset_name, split_name)]
        dataset = tf.data.TFRecordDataset(filenames)

        # for version 1.5 and above use tf.data.TFRecordDataset

        # example proto decode
        def _parse_function(example_proto):
            keys_to_features = {'X': tf.FixedLenFeature(([64, 64, 3]), tf.float32),
                                'y': tf.FixedLenFeature((), tf.int64, default_value=0)}
            parsed_features = tf.parse_single_example(example_proto, keys_to_features)

            return parsed_features['X'], parsed_features['y']

        # Parse the record into tensors.
        dataset = dataset.map(_parse_function)

        # Shuffle the dataset
        dataset = dataset.shuffle(buffer_size=10000)

        # Repeat the input indefinitly
        #dataset = dataset.repeat()

        # Generate batches
        dataset = dataset.batch(batch_size)

        # Create a one-shot iterator
        iterator = dataset.make_initializable_iterator()
        next_batch = iterator.get_next()


        '''
        dataset = self.get_split(split_name)
        provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset,
            common_queue_capacity=5 * batch_size,
            common_queue_min=batch_size,
            shuffle=(split_name == split_name))
        [image, label] = provider.get(['image', 'label'])
        # Preprocess the images.
        image = (tf.to_float(image)/127.5) - 1.
        # Creates a QueueRunner for the pre-fetching operation.
        images, labels = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=1,
            capacity=100 * batch_size)
            
        labels = tf.reshape(labels, [-1])
        if one_hot:
            labels = tf.one_hot(labels, dataset.num_classes)
        return images, labels, dataset.num_samples, dataset.num_classes, next_element, iterator
        '''
        return next_batch, iterator

    def float_image_to_uint8(self, image):
        image = (image * 127.5) + 127.5
        return tf.cast(image, tf.uint8)

    def get_total_number_of_images(self, split_name, dataset_name):
        file_pattern = dataset_name + '_' + split_name + '.tfrecord'
        file_pattern = os.path.join(self.dataset_storage_location, file_pattern)
        return sum(1 for _ in tf.python_io.tf_record_iterator(file_pattern))

    def get_tfrecord_file_location(self, dataset_name, split_name):
        file_pattern = dataset_name + '_' + split_name + '.tfrecord'
        return os.path.join(self.dataset_storage_location, file_pattern)

    def get_split(self, split_name):
        file_pattern = self.dataset_name+'_'+split_name+'.tfrecord'
        file_pattern = os.path.join(self.dataset_storage_location, file_pattern)
        print("file path.."+ file_pattern)
        reader = tf.TFRecordReader

        keys_to_features = {
            'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
            'image/format': tf.FixedLenFeature((), tf.string, default_value='png'),
            'image/class/label': tf.FixedLenFeature(
                [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
        }

        items_to_handlers = {
            'image': slim.tfexample_decoder.Image(shape=[self.image_size, self.image_size, self.channels]),
            'label': slim.tfexample_decoder.Tensor('image/class/label'),
        }

        decoder = slim.tfexample_decoder.TFExampleDecoder(
            keys_to_features, items_to_handlers)

        labels_to_names = None
        label_file_name = self.dataset_name + '_labels.txt'
        if dataset_utils.has_labels(self.dataset_storage_location, label_file_name):
            labels_to_names = dataset_utils.read_label_file(self.dataset_storage_location, label_file_name)

        total_number_of_images = self.get_total_number_of_images(split_name, self.dataset_name)
        return slim.dataset.Dataset(
            data_sources=file_pattern,
            reader=reader,
            decoder=decoder,
            items_to_descriptions=self.dataset_description,
            num_classes=self.number_of_classes,
            num_samples = total_number_of_images,
            labels_to_names=labels_to_names)



