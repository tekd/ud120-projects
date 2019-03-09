#!/usr/bin/python

"""
    This is the code t

    Use a Decision Tree to identify emails from the Enron corpus by author:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


# features_train and features_test are the features for the training
# and testing datasets, respectively
# labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


#########################################################
### your code goes here ###
from sklearn import tree
from sklearn.metrics import accuracy_score


def classify(features_train, labels_train, min_samples_split=2):
    clf = tree.DecisionTreeClassifier(min_samples_split=min_samples_split)
    clf.fit(features_train, labels_train)
    return clf


pred = classify(features_train, labels_train, 40).predict(features_test)

acc = accuracy_score(pred, labels_test)

print acc
# 0.9778156996587031

# get number of features
# print len(features_train[-1])  # 3785

# limit features
# selector = SelectPercentile(f_classif, percentile=1) in tools/email_preprocess.py
# 379

# acc with 1% of features
# 0.9670079635949943

#########################################################
