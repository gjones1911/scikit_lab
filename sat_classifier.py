import numpy as np
import pandas as pd
import sklearn as skl
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from DataManipulation import *
from performance_tests import get_best_performance
from DisplayMethods import display_plt_confu_mat
from DisplayMethods import show_best_params
from SVC_lib import grid_search_tester
from SVC_lib import z_normalize_svc
from sklearn.metrics import confusion_matrix
import itertools
from sklearn import svm

file_name_train = 'sat.trn'
file_name_test = 'sat.tst'

# use pandas to read the data into a data frame
sat_train_df = pd.read_csv(file_name_train, header=None, sep=' ')
sat_test_df = pd.read_csv(file_name_test, header=None, sep=' ')

# grab numpy version of the sets
np_train = sat_train_df.values
np_test = sat_test_df.values

# split up the sets into the attributes and classifications
train_set = [np.array(np_train[:, :-1].tolist(), dtype=np.float64),
             np.array(np_train[:, -1].tolist(), dtype=np.float64)]

validation_set = [np.array(np_test[0:2000, :-1].tolist(), dtype=np.float64),
                  np.array(np_test[0:2000, -1].tolist(), dtype=np.float64)]
test_set = [np.array(np_test[0:2000, :-1].tolist(), dtype=np.float64),
            np.array(np_test[0:2000:, -1].tolist(), dtype=np.float64)]

print('train data length')
print(len(train_set[0]))
print('Train class length')
print(len(train_set[1]))
print('val data length')
print(len(validation_set[0]))
print('val class length')
print(len(validation_set[1]))
print('test data length')
print(len(test_set[0]))
print('test class length')
print(len(test_set[1]))

# z normalize the cross validation sets
train_set, validation_set, test_set = z_normalize_svc(train_set, validation_set, test_set)

# Set the parameters by cross-validation
# -------------------------------   use parameters:
# {'C': 1, 'gamma': 0.1, 'kernel': 'rbf'} test only 91
# ---------------------------------------------------
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [.1, 1e-2, 1e-3, 1e-4, 1e-5],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['poly'], 'gamma':  [.1, 1e-2, 1e-3, 1e-4, 1e-5], 'degree': [2,3,8,9],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['sigmoid'], 'gamma': [.1, 1e-2, 1e-3, 1e-4, 1e-5],
                     'C': [1, 10, 100, 1000]}]

# -------------------------------   use parameters:
#  {'C': 8.0, 'gamma': 0.06, 'kernel': 'rbf'} val: 91, tst: 92 ()
# ---------------------------------------------------
tuned_parametersA = [{'kernel': ['rbf'], 'gamma': np.around(np.linspace(.01,.2,20).tolist(), 2),
                     'C': np.around(np.linspace(1, 10, 10).tolist(), 0)}]

# uses the best found parameters
tuned_parameters_Best = [{'kernel': ['rbf'], 'gamma': [.06],
                         'C': [8]}]

# -------------------------------   use parameters:
#  {'C': 7.4, 'gamma': 0.061, 'kernel': 'rbf'} .91, .92
# ---------------------------------------------------
# tuned_parametersB = [{'kernel': ['rbf'], 'gamma': np.around(np.linspace(.05,.07,20).tolist(), 3),
#                     'C': np.around(np.linspace(7.1, 9.1, 20).tolist(), 1)}]
# -------------------------------   use parameters:
#  {'C': 7.54, 'gamma': 0.0605, 'kernel': 'rbf'} .91, .92
# ---------------------------------------------------
# tuned_parametersC = [{'kernel': ['rbf'], 'gamma': np.around(np.linspace(.05,.07,200).tolist(), 4),
#                     'C': np.around(np.linspace(7, 8, 100).tolist(), 2)}]
# scoresA = ['precision_macro', 'accuracy']
# scores1 = ['precision_macro']
# scores2 = ['recall_macro']
# t_p = tuned_parametersB
# t_p = tuned_parametersC

t_p = tuned_parameters_Best
# t_p = tuned_parameters
# t_p = tuned_parametersA

# scores = ['precision_macro', 'recall_macro', 'accuracy']
scores3 = ['accuracy']

scr = scores3

labels = np.array([1,2,3,4,5,7], dtype=np.float64)
class_names = ['red soil', 'cotton crop', 'grey soil', 'damp grey soil',
               'soil with veg', 'very damp grey']

#clf, b_params, c_rV, c_rT, clf_l = grid_search_tester(train_set, validation_set, test_set, tuned_parameters5,
#                                                      scores=scores2, verbose=True, cv=5)
clf, b_params, c_rV, c_rT, clf_l = grid_search_tester(t_p, train_set, validation_set=None, test_set=test_set, cv=5,
                                                      scores=scr, verbose=True, show_param_test=False, t_names=labels)


show_best_params(b_params)
#b_avg, b_score = get_best_performance(c_rT, scr, labels, metric='precision')
#print('Should use score: {:s}'.format(b_score))
#print()
#print('param options: ')
#print(t_p)
#print('# -------------------------------   use parameters: ')
#print('# ', b_params[b_score])
#print('# ---------------------------------------------------')
#print()
#print('best score: ', b_score)
#print('best avg: ', b_avg)



#display_plt_confu_mat(clf, test_set, labels=labels, class_names=class_names)

show = False
for i in range(len(clf_l)):
    #display_plt_confu_mat(clf_l[i], validation_set, labels=labels, class_names=class_names, show_it=show,
    #                      data_name='Validation')

    if i == len(clf_l)-1:
        show=True
    display_plt_confu_mat(clf_l[i], test_set, labels=labels, class_names=class_names, show_it=show,
                          data_name='Test')