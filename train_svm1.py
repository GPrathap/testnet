from time import time
from sklearn.metrics import accuracy_score
from sklearn import svm, linear_model
import numpy as np
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
    #clf = svm.SVC(kernel='linear', C=C).fit(X_train, y_train)
    clf = linear_model.SGDClassifier(warm_start=True, max_iter=6).fit(X_train, y_train)
    print("done in %0.3fs" % (time() - t0))

    print("Hard mining")
    y_pred = clf.predict(X_train)
    difference = np.where((y_pred-y_train) !=0)
    hard_y_train = y_train[difference]
    hard_x_train = X_train[difference]
    clf = clf.fit(hard_x_train, hard_y_train)
    print("Predicting...")
    t0 = time()
    y_pred = clf.predict(X_test)

    print ("Accuracy: %.3f" % (accuracy_score(y_test, y_pred)))
    acc.append(accuracy_score(y_test, y_pred))
print (acc)
np.save('{}/accuracy_scores.npy'.format(FLAGS.feature_dir), acc)