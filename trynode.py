"""
Play with saving .
Closest:
    https://github.com/tensorflow/tensorflow/issues/616#issuecomment-205620223
"""
import numpy as np

import tensorflow as tf
from tensorflow.python.platform import gfile
# from tensorflow.python.training.training_util import write_graph
# from tensorflow.python.tools.freeze_graph import freeze_graph
from tensorflow.python.framework.graph_util import convert_variables_to_constants



def load_graph():

    checkpoint_path = tf.train.latest_checkpoint("/data/satellitegpu/train_log5/")
    saver = tf.train.import_meta_graph(checkpoint_path + ".meta")
    #saver = tf.train.import_meta_graph('/data/satellitegpu/train_log4/model.ckpt-98741.data-00000-of-00001')
    #new_saver.restore(sess, 'my-save-dir/my-model-10000')
    saver = tf.train.Saver()

    with tf.Session() as sess:
        #new_saver = tf.train.import_meta_graph('/data/satellitegpu/train_log4/model.ckpt-98741.data-00000-of-00001')
        #new_saver.restore(sess, '/data/satellitegpu/train_log4/model.ckpt-98741')
        saver.restore(sess, "model.ckpt-99230.data-00000-of-00001")
        print("---")
        #saver.restore(sess, checkpoint_path)
        #input_ = tf.get_collection("input:0", scope="")[0]
        #output_ = tf.get_collection("output:0", scope="")[0]

        #output = sess.run(output_, feed_dict={input_: np.arange(10, dtype=np.float32)})
        #print ("output", output)



        #freeze_graph(sess)


if __name__ == '__main__':
    import sys

    load_graph()
    #load_frozen_graph()