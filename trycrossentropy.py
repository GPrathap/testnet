import tensorflow as tf


p = tf.placeholder(tf.float32, shape=[None, 5])
logit_q = tf.placeholder(tf.float32, shape=[None, 5])
q = tf.nn.sigmoid(logit_q)

with tf.Session() as sess:

    feed_dict = {
      p: [[0, 0, 0, 1, 0],
          [1, 0, 0, 0, 0]],
      logit_q: [[0.2, 0.2, 0.2, 0.2, 0.2],
                [0.3, 0.3, 0.2, 0.1, 0.1]]
    }

    prob1 = -p * tf.log(q)
    prob2 = p * -tf.log(q) + (1 - p) * -tf.log(1 - q)
    prob3 = p * -tf.log(tf.sigmoid(logit_q)) + (1-p) * -tf.log(1-tf.sigmoid(logit_q))
    prob4 = tf.nn.sigmoid_cross_entropy_with_logits(labels=p, logits=logit_q)
    print(prob1.eval(feed_dict))
    print(prob2.eval(feed_dict))
    print(prob3.eval(feed_dict))
    print(prob4.eval(feed_dict))