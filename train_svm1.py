from time import time
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib
import tensorflow as tf

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
flags.DEFINE_string("checkpoint_dir", "/data/checkpoint43", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_string("feature_dir", "/data/features", "Directory name to save features")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("is_crop", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
flags.DEFINE_string('dataset_dir', '/data/satellitegpu/', 'Location of data.')
flags.DEFINE_string('dataset_path_train', '/data/images/uc_train_256_data/**.jpg', 'Location of training images data.')
flags.DEFINE_string('dataset_path_test', '/data/images/uc_test_256/**.jpg', 'Location of testing images data.')
flags.DEFINE_string('dataset_storage_location', '/data/neotx', 'Location of image store')
flags.DEFINE_string('dataset_name', 'ucdataset', 'Data set name')
FLAGS = flags.FLAGS

from sklearn import svm
#
import numpy as np

acc = []
nums = np.load('{}/features.npy'.format(FLAGS.feature_dir))

for num in nums:
    X_train = np.load('/data/features/features%d_train.npy' % num)
    y_train = np.load('/data/features/label%d_train.npy' % num)
    X_test = np.load('/data/features/features%d_test.npy' % num)
    y_test = np.load('/data/features/label%d_test.npy' % num)

    print("Fitting the classifier to the training set")
    t0 = time()
    C = 1000.0  # SVM regularization parameter
    clf = svm.SVC(kernel='linear', C=C).fit(X_train, y_train)
    print("done in %0.3fs" % (time() - t0))

    print("Predicting...")
    t0 = time()
    y_pred = clf.predict(X_test)

    print ("Accuracy: %.3f" % (accuracy_score(y_test, y_pred)))
    acc.append(accuracy_score(y_test, y_pred))
print (acc)
np.save('{}/accuracy_scores.npy'.format(FLAGS.feature_dir), acc)