#!/usr/bin/python

from numpy import mean
import sklearn
from sklearn import cross_validation
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.cross_validation import train_test_split

import sys

def evaluate_clf(clf, features, labels,num_iters=1000,test_size=0.3, random_state=42):
    print clf
    accuracy = []
    precision = []
    recall = []
    first = True
    #sss= sklearn.cross_validation.StratifiedShuffleSplit(labels, n_iter=3,test_size=0.5, random_state=0)
    #print sss
    #for train_index,test_index in sss:
    	#print("TRAIN:",train_index, "TEST:",test_index)
	#print labels[test_index]
	#features_train,features_test = features[train_index],features[test_index]
	#labels_train,labels_test = labels[train_index],labels[test_index]
    for trial in range(num_iters):
        features_train, features_test, labels_train, labels_test =\
            cross_validation.train_test_split(features, labels, test_size=test_size)
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        accuracy.append(accuracy_score(labels_test, predictions))
        precision.append(precision_score(labels_test, predictions))
        recall.append(recall_score(labels_test, predictions))
        #if trial % 10 == 0:
        #    if first:
        #        sys.stdout.write('\nProcessing')
        #    sys.stdout.write('.')
        #    sys.stdout.flush()
        #    first = False

    #print "done.\n"
    #print "accuracy: {}".format(accuracy)
    print "precision: {}".format(mean(precision))
    print "recall:    {}".format(mean(recall))
    return mean(precision), mean(recall)
