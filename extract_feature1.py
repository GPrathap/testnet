import os
import pprint

from convert_to_tf_record import DataConvertor
from network import *

pp = pprint.PrettyPrinter()

from Config import *

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
    net_d, d_logits, features = neoxt.discriminator(real_images, is_train=False,
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
            feature_ids = []
            print("Start processing on {} dataset".format(type))
            while True:
                try:
                    next_batch_ = sess.run([next_batch])
                    batch_images = np.array(next_batch_[0][0], dtype=np.float32) / 127.5 - 1
                    batch_labels = np.array(next_batch_[0][1], dtype=np.int)
                    #batch_ids = np.array(next_batch_[0][2], dtype=np.str)
                    feat = sess.run(features, feed_dict={real_images: batch_images})
                    feature_vectors.append(feat)
                    feature_labels.append(batch_labels)
                    #feature_ids.append(batch_images)

                except tf.errors.OutOfRangeError:
                    feature_vectors = np.array(feature_vectors)
                    feature_vectors = feature_vectors.reshape(-1, feature_vectors.shape[2])

                    feature_labels = np.array(feature_labels)
                    feature_labels = feature_labels.reshape(-1)

                    feature_ids = np.array(feature_ids)
                    feature_ids = feature_ids.reshape([-1, FLAGS.image_size, FLAGS.image_size, FLAGS.c_dim])

                    name_for_features = '{}/features{}_{}.npy'.format(FLAGS.feature_dir, num, type)
                    name_for_feature_labels = '{}/label{}_{}.npy'.format(FLAGS.feature_dir, num, type)
                    #name_for_feature_ids = '{}/feature_ids{}_{}.npy'.format(FLAGS.feature_dir, num, type)

                    np.save(name_for_features, feature_vectors)
                    np.save(name_for_feature_labels, feature_labels)
                    #np.save(name_for_feature_ids, feature_ids)

                    break
    np.save('{}/features.npy'.format(FLAGS.feature_dir), checkpoints_numbers)

if __name__ == '__main__':
    tf.app.run()
