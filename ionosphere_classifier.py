import numpy as np
import pandas as pd
import sklearn as skl
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from performance_tests import get_best_performance
from DataManipulation import *
from DisplayMethods import show_best_params
from DisplayMethods import display_plt_confu_mat
from SVC_lib import grid_search_tester
from SVC_lib import z_normalize_svc
from sklearn.metrics import confusion_matrix
import itertools
from sklearn import svm

file_name = 'ionosphere.data'

# use pandas to read the data into a data frame
ionosphere_df = pd.read_csv(file_name, header=None)

# remove the column of zero's
ionosphere_df.drop(1, axis=1, inplace=True)

# generate randomized training validation and test sets from data frame
train_set, validation_set, test_set, ks_ds = df_panda_splitter(ionosphere_df, seed=True, normalize=None)

# z normalize the cross validation sets
train_set, validation_set, test_set = z_normalize_svc(train_set, validation_set, test_set)

# Set the parameters by cross-validation
#tuned_parameters = [{'kernel': ['rbf'], 'gamma': [.1, 1e-2, 1e-3, 1e-4, 1e-5],
#                     'C': [1, 10, 100, 1000]},
#                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

# -------------------------------   use parameters:
#  {'C': 10, 'coef0': 0, 'gamma': 0.1, 'kernel': 'rbf'}
# ---------------------------------------------------
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [.1, 1e-2, 1e-3, 1e-4, 1e-5], 'coef0': [0,.5,1,5,10],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['poly'], 'gamma':  [.1, 1e-2, 1e-3, 1e-4, 1e-5], 'degree': [2,3,4,5],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['sigmoid'], 'gamma': [.1, 1e-2, 1e-3, 1e-4, 1e-5],
                     'C': [1, 10, 100, 1000]}]

# -------------------------------   use parameters:
#  {'C': 4.0, 'gamma': 0.04, 'kernel': 'rbf'} score: precision macro
# ---------------------------------------------------
tuned_parameters2b = [{'kernel': ['rbf'], 'gamma': np.around(np.linspace(.01,.2,20).tolist(), 2),
                       'C': np.around(np.linspace(1, 101, 100).tolist(), 0)}]

tuned_parameters_Best = [{'kernel': ['rbf'], 'gamma': [.04], 'C': [4]}]

scores = ['precision_macro', 'recall_macro', 'balanced_accuracy']
scores2 = ['precision_macro']

scr = scores2

# t_p = tuned_parameters         # *****
# t_p = tuned_parameters2b
t_p = tuned_parameters_Best

labels = ['g', 'b']

clf, b_params, c_rV, c_rT, clf_l = grid_search_tester(t_p, train_set, validation_set, test_set, cv=5, scores=scr,
                                                      verbose=True, show_param_test=False, t_names=labels)

show_best_params(b_params, c_rV, c_rT)

b_avg, b_score = get_best_performance(c_rT, scr, labels, metric='precision')

print('Should use score: {:s}'.format(b_score))
print()
print('param options: ')
print(t_p)
print('# -------------------------------   use parameters: ')
print('# ', b_params[b_score])
print('# ---------------------------------------------------')
print()
print('best score: ', b_score)
print('best avg: ', b_avg)

show=False
for i in range(len(clf_l)):
    display_plt_confu_mat(clf_l[i], validation_set, labels=['g','b'], class_names=['g','b'], show_it=show,
                          data_name='Validation data')
    if i == len(clf_l)-1:
        show=True
    display_plt_confu_mat(clf_l[i], test_set, labels=['g','b'], class_names=['g','b'], show_it=show,
                          data_name='Test data')
