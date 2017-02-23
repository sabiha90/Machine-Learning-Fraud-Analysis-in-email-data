#!/usr/bin/python
import csv
import matplotlib.pyplot
import pickle
import sys
from math import isnan
from sklearn.feature_selection import SelectKBest
sys.path.append("../tools/")
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.cross_validation import StratifiedKFold
from feature_format import featureFormat
from feature_format import targetFeatureSplit


###finds the outliers from the data
def find_outlier(data_dict):
    counts = dict.fromkeys(data_dict.itervalues().next().keys(), 0)
    for record in data_dict:
        person = data_dict[record]
        for field in person:
            if person[field] == 'NaN':
                continue
	#print record
    

###removes outliers
def remove_outliers(dict_object, keys):
    for key in keys:
        dict_object.pop(key, 0)



###Counts the valid non NaN values
def count_valid_values(data_dict):
    """ counts the number of non-NaN values for each feature """
    counts = dict.fromkeys(data_dict.itervalues().next().keys(), 0)
    for record in data_dict:
        person = data_dict[record]
        for field in person:
            if person[field] != 'NaN':
                counts[field] += 1
    print counts

###Visualizing data points against poi using a scatter plot
def visualize(data_dict, feature_x, feature_y):

    data = featureFormat(data_dict, [feature_x, feature_y, 'poi'])

    for point in data:
        x = point[0]
        y = point[1]
        poi = point[2]
        color = 'red' if poi else 'blue'
        matplotlib.scatter(x, y, color=color)
    matplotlib.xlabel(feature_x)
    matplotlib.ylabel(feature_y)
    matplotlib.show()


###Selecting the k best features from the dataset using sklearn's SelectKBest
def get_k_best(data_dict, features_list, k):
    data = featureFormat(data_dict, features_list)
    labels, features = targetFeatureSplit(data)

    k_best = SelectKBest(k=k)
    k_best.fit(features, labels)
    scores = k_best.scores_
    unsorted_pairs = zip(features_list[1:], scores)
    sorted_pairs = list(reversed(sorted(unsorted_pairs, key=lambda x: x[1])))
    k_best_features = dict(sorted_pairs[:k])
    print "{0} best features: {1} - {2}\n".format(k, k_best_features.keys(),k_best_features.values())
    return k_best_features

###Creating two new features
###The total number of emails to and from a POI divided by the total number of emails received

def add_poi_email_interaction(data_dict, features_list):
    fields = ['to_messages', 'from_messages',
              'from_poi_to_this_person', 'from_this_person_to_poi']
    for record in data_dict:
        person = data_dict[record]
        is_valid = True
        for field in fields:
            if person[field] == 'NaN':
                is_valid = False
        if is_valid:
            total_messages = person['to_messages'] +\
                             person['from_messages']
            poi_messages = person['from_poi_to_this_person'] +\
                           person['from_this_person_to_poi']
            person['email_interaction'] = float(poi_messages) / total_messages
        else:
            person['email_interaction'] = 'NaN'
    features_list += ['email_interaction']

###The sum of exercised_stock_options, salary and total_stock_value that reveals the total wealth a POI had.

def add_financial_aggregate(data_dict, features_list):
    """ mutates data dict to add aggregate values from stocks and salary """
    fields = ['total_stock_value', 'exercised_stock_options', 'salary']
    for record in data_dict:
        person = data_dict[record]
        is_valid = True
        for field in fields:
            if person[field] == 'NaN':
                is_valid = False
        if is_valid:
            person['financial_aggregate'] = sum([person[field] for field in fields])
        else:
            person['financial_aggregate'] = 'NaN'
    features_list += ['financial_aggregate']


def stratified_k_fold(clf,features,labels):
    skf = StratifiedKFold( labels, n_folds=3 )
    precisions = []
    recalls = []
    for train_idx, test_idx in skf:
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )

        ### fit the classifier using training set, and test on test set
        clf.fit(features_train, labels_train)
        pred = clf.predict(features_test)


        ### for each fold, print some metrics
        print
        print "precision score: ", precision_score( labels_test, pred )
        print "recall score: ", recall_score( labels_test, pred )

        precisions.append( precision_score(labels_test, pred) )
        recalls.append( recall_score(labels_test, pred) )

    ### aggregate precision and recall over all folds
    print "average precision: ", sum(precisions)/3.
    print "average recall: ", sum(recalls)/3.




if __name__ == '__main__':
    data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
    outliers = ['TOTAL', 'THE TRAVEL AGENCY IN THE PARK', 'LOCKHART EUGENE E']
    #remove_keys(data_dict, outliers)
    find_outlier(data_dict)
    #make_csv(data_dict)
    #visualize(data_dict, 'salary', 'bonus')
    # visualize(data_dict, 'from_poi_to_this_person', 'from_this_person_to_poi')
    count_valid_values(data_dict)
    features_list = ['poi',
                     'bonus',
                     'deferral_payments',
                     'deferred_income',
                     'director_fees',
                     'exercised_stock_options',
                     'expenses',
                     'loan_advances',
                     'long_term_incentive',
                     'other',
                     'restricted_stock',
                     'restricted_stock_deferred',
                     'salary',
                     'total_payments',
                     'total_stock_value',
                     'from_messages',
                     'from_poi_to_this_person',
                     'from_this_person_to_poi',
                     'shared_receipt_with_poi',
                     'to_messages']

    #add_email_interaction(data_dict, features_list)
    #visualize(data_dict, 'total_stock_value', 'email_interaction')

    #add_financial_aggregate(data_dict, features_list)
    #visualize(data_dict, 'email_interaction', 'financial_aggregate')

    #k_best = get_k_best(data_dict, features_list, 10)
    #features = ['email_interaction','financial_aggregate']
    #k_best = get_k_best(data_dict, features,2)
    #print k_best
