import tensorflow as tf

# Let's load a previously saved meta graph in the default graph
# This function returns a Saver
saver = tf.train.import_meta_graph('/data/satellitegpu/train_log5/model.ckpt-35435.meta')

# We can now access the default graph where all our metadata has been loaded
graph = tf.get_default_graph()

with tf.Session() as sess:
    # To initialize values with saved data
    saver.restore(sess, '/data/satellitegpu/train_log5/model.ckpt-35435')
