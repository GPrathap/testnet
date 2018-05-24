from time import time
from sklearn.metrics import accuracy_score
from sklearn import svm
import numpy as np
from Config import *

acc = []
nums = np.load('{}/features.npy'.format(FLAGS.feature_dir))

for num in nums:
    X_train = np.load('{}/features%d_train.npy'.format(FLAGS.feature_dir, num))
    y_train = np.load('{}/label%d_train.npy'.format(FLAGS.feature_dir, num))
    X_test = np.load('{}/features%d_test.npy'.format(FLAGS.feature_dir, num))
    y_test = np.load('{}/label%d_test.npy'.format(FLAGS.feature_dir, num))

    print("Fitting the classifier to the training set")
    t0 = time()
    C = 1000.0
    clf = svm.SVC(kernel='linear', C=C).fit(X_train, y_train)
    print("done in %0.3fs" % (time() - t0))

    print("Predicting...")
    t0 = time()
    y_pred = clf.predict(X_test)

    print ("Accuracy: %.3f" % (accuracy_score(y_test, y_pred)))
    acc.append(accuracy_score(y_test, y_pred))
print (acc)
np.save('{}/accuracy_scores.npy'.format(FLAGS.feature_dir), acc)