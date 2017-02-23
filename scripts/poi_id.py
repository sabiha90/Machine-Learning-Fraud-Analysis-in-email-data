#!/usr/bin/python
from copy import copy
import sys
import pickle
import sklearn
sys.path.append("../tools/")

from sklearn.model_selection import GridSearchCV
from feature_format import featureFormat, targetFeatureSplit
import data_cleaner
import evaluation

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
target_label = 'poi'
email_features_list = [
	'from_messages',
	'from_poi_to_this_person',
	'from_this_person_to_poi',
	'shared_receipt_with_poi',
	'to_messages',
	]
financial_features_list = [
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
]
features_list = [target_label] + financial_features_list + email_features_list
### Load the dictionary containing the dataset
data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### Task 2: Remove outliers
outlier_keys = ['TOTAL', 'THE TRAVEL AGENCY IN THE PARK', 'LOCKHART EUGENE E']
data_cleaner.remove_outliers(data_dict, outlier_keys)


### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = copy(data_dict)
my_feature_list = copy(features_list)

# Get K-best features
num_features = 10
best_features = data_cleaner.get_k_best(my_dataset, my_feature_list, num_features)
my_feature_list = [target_label] + best_features.keys()
#print "selected:",my_feature_list[1:]

# Add two new features
data_cleaner.add_financial_aggregate(my_dataset, my_feature_list)
data_cleaner.add_poi_email_interaction(my_dataset, my_feature_list)

#num_features = 6
#best_features = data_cleaner.get_k_best(my_dataset, my_feature_list, num_features)
#print best_features
#my_features_list = [target_label] + best_features.keys()+['email_interaction','financial_aggregate']
#print my_features_list
#print "{0} selected features: {1}\n".format(len(my_features_list), my_features_list[1:])



#Print features
print "{0} selected features: {1}\n".format(len(my_feature_list) - 1, my_feature_list[1:])

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, my_feature_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


# Scale features using MinMaxScaler
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
features = scaler.fit_transform(features)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
features_train,features_test,labels_train,labels_test = sklearn.cross_validation.train_test_split(features,labels, test_size=0.3, random_state=42)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Decision Tree",
         "Random Forest", "AdaBoost", "Naive Bayes"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="rbf", C=1000),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    #AdaBoostClassifier(algorithm='SAMME'),
    GaussianNB()]
for name, clf in zip(names, classifiers):
        clf.fit(features_train,labels_train)
        scores = clf.score(features_test,labels_test)
        print " "
        print "Classifier:"
        evaluation.evaluate_clf(clf, features, labels, num_iters=1000, test_size=0.3,random_state=42)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        print "====================================================================="

#params = {'kneighborsclassifier-1__n_neighbors': [1, 5],
#          'kneighborsclassifier-2__n_neighbors': [1, 5],
#          'randomforestclassifier__n_estimators': [10, 50]}

#precision,recall = evaluation.evaluate_clf(clf,features,labels)
#scoring=[precision,recall]
#cv = sklearn.cross_validation.StratifiedShuffleSplit(labels, n_iter=10)
#for score in scoring:
#    print("# Tuning hyper-parameters for %s" % score)
#    print()
#
#    clf = GridSearchCV(classifiers, params, cv=5,
#                       scoring='%s_macro' % score)
#    clf.fit(features_train, labels_train)
#   print("Best parameters set found on development set:")
#    print()
#    print(clf.best_params_)

gauss_clf = GaussianNB(priors=None)    
d_clf = DecisionTreeClassifier(max_depth=5)
### Final Machine Algorithm Selection
clf = gauss_clf


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!

evaluation.evaluate_clf(gauss_clf, features, labels)
#evaluation.evaluate_clf(d_clf, features, labels)
data_cleaner.stratified_k_fold(gauss_clf,features,labels)
#sss= sklearn.cross_validation.StratifiedShuffleSplit(labels, n_iter=3,test_size=0.5, random_state=0)
##print sss
#for train_index,test_index in sss:
#    print("TRAIN:",train_index, "TEST:",test_index)
#    print labels[test_index]
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

pickle.dump(clf, open("../final_project/my_classifier.pkl", "w"))
pickle.dump(my_dataset, open("../final_project/my_dataset.pkl", "w"))
pickle.dump(my_feature_list, open("../final_project/my_feature_list.pkl", "w"))

