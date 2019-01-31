import pandas as pd
from DataExplorerPD import *
from DataManipulation import *
from DisplayMethods import show_best_params
from performance_tests import get_best_performance
from DisplayMethods import display_plt_confu_mat
from DisplayMethods import show_best_params
from SVC_lib import grid_search_tester
from SVC_lib import z_normalize_svc
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from DisplayMethods import plot_confusion_matrix

# get the name of my modified file of vowel context data
# i cleaned up the file to remove some double spaces that confused
# the pandas reac_csv function
# the load the data from modified file
vowel_file = 'vowel-context.dt'
vowel_context = pd.read_csv(vowel_file, header=None, sep=' ')

# remove irrelevant attributes
vowel_context.drop(columns=[0,1,2], inplace=True)

# generate randomized training validation and test sets from data frame
train_set, validation_set, test_set, ks_ds = df_panda_splitter(vowel_context, seed=True, normalize=None)

# z normalize the cross validation sets
train_set, validation_set, test_set = z_normalize_svc(train_set, validation_set, test_set)

# ########################################################################################################
# ########################################################################################################
# ########################################################################################################
# ##############################################  set of tuning parameters  ##############################
# Set the parameters by cross-validation
# parameters for course grid search

# -------------------------------   use parameters:
#  {'C': 100, 'gamma': 0.1, 'kernel': 'rbf'}
# ---------------------------------------------------
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [.1, 1e-2, 1e-3, 1e-4, 1e-5], 'coef0': [0,.5,1,5,10],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['poly'], 'gamma':  [.1, 1e-2, 1e-3, 1e-4, 1e-5], 'degree': [2,3,6,8],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['sigmoid'], 'gamma': [.1, 1e-2, 1e-3, 1e-4, 1e-5],
                     'C': [1, 10, 100, 1000]}]

# #######################################################################
# ########################################### C tuning
# #######################################################################

# -------------------------------   use parameters:
#  {'C': 100.0, 'gamma': 0.17, 'kernel': 'rbf'} 99 100
# ---------------------------------------------------
tuned_parametersA = [{'kernel': ['rbf'], 'gamma': np.around(np.linspace(.01,.2,20).tolist(), 2),
                     'C': np.around(np.linspace(100, 1000, 900).tolist(), 0)}]

tuned_parameters_Best = [{'kernel': ['rbf'], 'gamma': [.17],
                          'C': [100]}]

scores = ['precision_macro', 'recall_macro', 'accuracy', 'f1_macro', 'f1_micro']
scores3 = ['accuracy']


# scoresAll = ['accuracy', 'precision_macro', 'recall', 'brier_score_loss', 'f1', 'roc_auc']
scoresAll = ['f1_macro', 'f1_micro']

labels = np.array([0,1,2,3,4,5,6,7,8,9,10], dtype=np.float64)
class_names = ['hid', 'hId', 'hEd', 'hAd', 'hYd', 'had', 'hOd', 'hod', 'hUd', 'hud', 'hed']

#t_p = tuned_parameters
#t_p = tuned_parametersA
t_p = tuned_parameters_Best

scr = scores3
clf, b_params, c_rV, c_rT, clf_l = grid_search_tester(t_p, train_set, validation_set, test_set, cv=10, scores=scr,
                                                      verbose=True, show_param_test=False, t_names=labels)


show_best_params(b_params)
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



#display_plt_confu_mat(clf, test_set, labels=labels, class_names=class_names)
show=False
for i in range(len(clf_l)):
    display_plt_confu_mat(clf_l[i], validation_set, labels=labels, class_names=class_names, show_it=show,
                          data_name=scr[i]+' V')
    if i == len(clf_l)-1:
        show=True
    display_plt_confu_mat(clf_l[i], test_set, labels=labels, class_names=class_names, show_it=show,
                          data_name=scr[i]+' T')

