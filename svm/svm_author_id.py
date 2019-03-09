#!/usr/bin/python

"""
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:
    Sara has label 0
    Chris has label 1
    """

import sys
import numpy as np

from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# features_train and features_test are the features for the training
# and testing datasets, respectively
# labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


#########################################################

# train on a smaller set of data
# features_train = features_train[:len(features_train) / 100]
# labels_train = labels_train[:len(labels_train) / 100]

# clf = SVC(kernel="linear")
clf = SVC(kernel="rbf", C=10000.0)
# fit the data
t0 = time()
clf.fit(features_train, labels_train)
print "training time: ", round(time() - t0, 3), "s"
# store predictions (returns labels)
t1 = time()
pred = clf.predict(features_test)
print "predict time: ", round(time() - t1, 3), 's'

# get the accuracy by comparing the prediction to the labels test
print accuracy_score(pred, labels_test)
print 'answer 10', pred[10]
print 'answer 26', pred[26]
print 'answer 50', pred[50]

predArr = np.array(pred)
print 'chris', np.count_nonzero(predArr == 1)

# predict time:  24.489 s
# 0.9908987485779295
# answer 10 1
# answer 26 0
# answer 50 1
# chris 877

#########################################################
