from time import time

from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
from sklearn import svm, linear_model
import numpy as np
from sklearn.multiclass import OneVsRestClassifier

from Config import *

acc = []
nums = np.load('{}/features.npy'.format(FLAGS.feature_dir))

for num in nums:
    X_train = np.load('{}/features{}_train.npy'.format(FLAGS.feature_dir, num))
    y_train = np.load('{}/label{}_train.npy'.format(FLAGS.feature_dir, num))

    X_test = np.load('{}/features{}_test.npy'.format(FLAGS.feature_dir, num))
    y_test = np.load('{}/label{}_test.npy'.format(FLAGS.feature_dir, num))

    print("Fitting the classifier to the training set")
    t0 = time()
    C = 1000.0
    n_estimators = 14

    clf = OneVsRestClassifier(BaggingClassifier(svm.SVC(kernel='linear', C=C)
                                                , max_samples=1.0 / n_estimators, n_estimators=n_estimators), n_jobs=7)
    clf = clf.fit(X_train, y_train)
    print("done in %0.3fs" % (time() - t0))
    y_pred = clf.predict(X_test)
    print("Predicting on training ...")
    print("Accuracy: %.3f" % (accuracy_score(y_test, y_pred)))

    acc.append(accuracy_score(y_test, y_pred))

print (acc)
np.save('{}/accuracy_scores.npy'.format(FLAGS.feature_dir), acc)