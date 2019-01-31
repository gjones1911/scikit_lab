# from __future__ import print_function
import numpy as np
import pandas as pd
import sklearn as skl
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from DataManipulation import *


file_name = 'ionosphere.data'

ionosphere_df = pd.read_csv(file_name, header=None)

# print('original data frame:')
# print(ionosphere_df)


ionosphere_df.drop(1, axis=1, inplace=True)
'''
print('modified data frame:')
print(ionosphere_df)
print()
print()
print()
print()
'''
'''
#print('The data as a numpy array')
#print(ionosphere_df.values[:, 0:34])
#print()
'''

train_set, validation_set, test_set, ks_ds = df_panda_splitter(ionosphere_df, seed=True, normalize=None)

'''
print('Training attributes:')
print(train_set[0])
print('Training classifications:')
print(train_set[1].shape)
'''

# get a standardizer
scaler = StandardScaler()

# adjust the standardizer parameters
scaler.fit(train_set[0])

'''
print('scaler data')
print()
print('mean: ')
print(scaler.mean_)
print()
print('transformed data:')
print(scaler.transform(train_set[0]))
print('copied:')
'''

# standardize the training data set
train_set[0] = scaler.transform(train_set[0])

'''
print(train_set[0])
print()
print()
print('Validation set transformed:')
'''


# standardize the validation set
validation_set[0] = scaler.transform(validation_set[0])

'''
print(validation_set[0])
print()
print()
print('Test set transformed:')
'''

# standardize the test set
test_set[0] = scaler.transform(test_set[0])
#print(test_set[0])


#train_df = pd.DataFrame(train_set[0])

# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [.1, 1e-2, 1e-3, 1e-4, 1e-5],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

#scores = ['precision_macro', 'recall_macro', 'accuracy']

scores = ['precision_macro', 'accuracy', 'balanced_accuracy' ]

print('the options are: \n',sorted(skl.metrics.SCORERS.keys()))
for score in scores:
    print("# Tuning hyper-parameters for {:s}".format(score))
    print()

    clf = GridSearchCV(SVC(), tuned_parameters, cv=5,
                       scoring=score)
    clf.fit(train_set[0], train_set[1])

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print('----------------------------------')
        print('for params: ', params)
        print("mu: {:0.3f}, std: (+/-{:0.03f}) ".format(mean, std * 2))
        print('----------------------------------')
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full training set.")
    print("The scores are computed on the full validation set.")
    print()
    y_true, y_pred = validation_set[1], clf.predict(validation_set[0])
    print(classification_report(y_true, y_pred))
    print()

    print("Detailed classification report for testing:")
    print()
    print("The model is trained on the full Training set.")
    print("The scores are computed on the full Testing set.")
    print()
    y_true, y_pred = test_set[1], clf.predict(test_set[0])
    print(classification_report(y_true, y_pred))
    print()












'''
print('descriptions')
print(train_df.info())
print('training data frame')
print(train_df.isna())
'''