
from time import time

import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score

import os
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.utils.inspect_checkpoint import print_tensors_in_checkpoint_file

from Utils import get_init_vector
from cifar import networkssate as networks, data_provider_sattelite as data_provider, data_provider_sattelite
from cifar import dcgansatelite as dcgan
from cifar import util
from tensorflow.contrib import predictor, slim
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix
import itertools


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
flags.DEFINE_string('checkpoint_dir', '/data/checkpoint1',
                    'Directory where the model was written to.')

X_train = np.load(FLAGS.checkpoint_dir + '/features_train.npy')
y_train = np.load(FLAGS.checkpoint_dir + '/features_train_class_label.npy')
X_test = np.load(FLAGS.checkpoint_dir + '/features_test.npy')
y_test = np.load(FLAGS.checkpoint_dir + '/features_test_class_label.npy')

print("Fitting the classifier to the training set")
C = 1000.0  # SVM regularization parameter
clf = svm.SVC(kernel='linear', C=C).fit(X_train, y_train)
classesList = ["agricultural", 'airplane', 'baseballdiamond', 'beach', 'buildings', 'chaparral'
      , 'denseresidential', 'forest', 'freeway', 'golfcourse', 'harbor', 'intersection', 'mediumresidential',
                 'mobilehomepark', 'overpass', 'parkinglot', 'river', 'runway', 'sparseresidential',
                 'storagetanks', 'tenniscourt']


def _plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    #fig ,ax = plt.subplots(10, 10)
    plt.imshow(cm, interpolation='nearest', cmap=cmap, aspect="auto")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def plot_confusion_matrix(actual_value, predicted_values, classes):
    cnf_matrix = confusion_matrix(actual_value, predicted_values)
    np.set_printoptions(precision=2)

    plt.figure(figsize=(15, 10))
    _plot_confusion_matrix(cnf_matrix, classes=classes,
                          title='Confusion matrix, without normalization')

    plt.figure(figsize=(15, 10))
    _plot_confusion_matrix(cnf_matrix, classes=classes, normalize=True,
                          title='Normalized confusion matrix')

    plt.show()

print("Predicting...")
y_pred = clf.predict(X_test)
np.save(FLAGS.checkpoint_dir + "/features_predict_class_label.npy", y_pred)

y_pred = np.load(FLAGS.checkpoint_dir + "/features_predict_class_label.npy")
print("Accuracy: %.3f" % (accuracy_score(y_test, y_pred)))
acc = accuracy_score(y_test, y_pred)
plot_confusion_matrix(y_test, y_pred, classesList)
print(acc)


