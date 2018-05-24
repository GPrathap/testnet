import os
import pprint

from convert_to_tf_record import DataConvertor
from network import *

pp = pprint.PrettyPrinter()

flags = tf.app.flags
flags.DEFINE_integer("epoch", 200, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", 40000, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 16, "The number of batch images [64]")
flags.DEFINE_integer("image_size", 64, "The size of image to use (will be center cropped) [108]")
flags.DEFINE_integer("output_size", 64, "The size of the output images to produce [64]")
flags.DEFINE_integer("sample_size", 64, "The number of sample images [64]")
flags.DEFINE_integer("c_dim", 3, "Dimension of image color. [3]")
flags.DEFINE_integer("sample_step", 500, "The interval of generating sample. [500]")
flags.DEFINE_integer("save_step", 50, "The interval of saveing checkpoints. [500]")
flags.DEFINE_string("dataset", "uc_train_256_data", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("checkpoint_dir", "/data/checkpoint50", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_string("feature_dir", "/data/features50", "Directory name to save features")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("is_crop", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
flags.DEFINE_string('dataset_dir', '/data/satellitegpu/', 'Location of data.')
flags.DEFINE_string('dataset_path_train', '/data/images/uc_train_256_data/**.jpg', 'Location of training images data.')
flags.DEFINE_string('dataset_path_test', '/data/images/uc_test_256/**.jpg', 'Location of testing images data.')
flags.DEFINE_string('dataset_storage_location', '/data/neotx', 'Location of image store')
flags.DEFINE_string('dataset_name', 'ucdataset', 'Data set name')
FLAGS = flags.FLAGS



def main(_):
    pp.pprint(flags.FLAGS.__flags)

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

    real_images =  tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.output_size, FLAGS.output_size,
                                               FLAGS.c_dim], name='real_images')

    neoxt = Neotx()
    # z --> generator for training
    net_d, d_logits, features = neoxt.discriminator(real_images, is_train=FLAGS.is_train,
                                                             reuse=False)
    data_convotor = DataConvertor(FLAGS.image_size, FLAGS.dataset_name,
                                  FLAGS.dataset_storage_location, FLAGS.c_dim)
    next_batch_train, iterator_train = data_convotor.provide_data(FLAGS.batch_size, 'train')
    next_batch_test, iterator_test = data_convotor.provide_data(FLAGS.batch_size, 'test')

    next_batch_list = [next_batch_train, next_batch_test]
    next_batch_iterator_list = [iterator_train, iterator_test]
    next_batch_iterator_type = ["train", "test"]

    saver = tf.train.Saver()
    sess=tf.Session()
    sess.run(tf.initialize_all_variables())

    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    checkpoints_numbers = []
    for checkpoint_path in ckpt.all_model_checkpoint_paths:
        num = int(checkpoint_path.split("/")[-1].replace("model-",""))
        checkpoints_numbers.append(num)
        saver.restore(sess, checkpoint_path)
        print("Loading checkpoints... {}".format(checkpoint_path))

        for iterator, next_batch, type in zip(next_batch_iterator_list, next_batch_list, next_batch_iterator_type):
            sess.run(iterator.initializer)
            if not os.path.exists(FLAGS.feature_dir):
                os.makedirs(FLAGS.feature_dir)
            feature_vectors = []
            feature_labels = []
            print("Start processing on {} dataset".format(type))
            while True:
                try:
                    next_batch_ = sess.run([next_batch])
                    batch_images = np.array(next_batch_[0][0], dtype=np.float32) / 127.5 - 1
                    batch_labels = np.array(next_batch_[0][1], dtype=np.int)
                    feat = sess.run(features, feed_dict={real_images: batch_images})
                    feature_vectors.append(feat)
                    feature_labels.append(batch_labels)

                except tf.errors.OutOfRangeError:
                    feature_vectors = np.array(feature_vectors)
                    feature_vectors = feature_vectors.reshape(-1, feature_vectors.shape[2])

                    feature_labels = np.array(feature_labels)
                    feature_labels = feature_labels.reshape(-1)
                    name_for_features = '{}/features{}_{}.npy'.format(FLAGS.feature_dir, num, type)
                    name_for_feature_labels = '{}/label{}_{}.npy'.format(FLAGS.feature_dir, num, type)
                    np.save( name_for_features, feature_vectors)
                    np.save(name_for_feature_labels, feature_labels)
                    break
    np.save('{}/features.npy'.format(FLAGS.feature_dir), checkpoints_numbers)

if __name__ == '__main__':
    tf.app.run()
