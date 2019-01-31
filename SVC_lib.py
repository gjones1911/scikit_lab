import numpy as np
import pandas as pd
import sklearn as skl
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from DataManipulation import *
from sklearn.preprocessing import StandardScaler

# ##########################################################################################################
# ##########################################################################################################
# ##########################################################################################################
# ######################################   tuneing functions   #############################################


def z_normalize_svc(train_set, validation_set, test_set):
    # get a standardizer
    scaler = StandardScaler()

    # adjust the standardizer parameters
    scaler.fit(train_set[0])

    # standardize the training data set
    train_set[0] = scaler.transform(train_set[0])

    # standardize the validation set
    validation_set[0] = scaler.transform(validation_set[0])

    # standardize the test set
    test_set[0] = scaler.transform(test_set[0])

    return train_set, validation_set, test_set


def grid_search_tester(tuned_parameters, train_set, validation_set=None, test_set=None, cv=5, scores=None, verbose=True,
                       show_param_test=True, t_names=None, num_classes=2):

    if t_names is None:
        t_names = []
        for i in range(num_classes):
            t_names.append('class ' + str(i))

    b_params = {}
    c_reportsV = {}
    c_reportsT = {}
    clf = None
    clf_l = []

    if scores is None:
        scores = ['precision_macro', 'recall_macro', 'accuracy']
    else:
        print('the scores are \n', scores)
        print()
    if verbose:
        print('the options are: \n', sorted(skl.metrics.SCORERS.keys()))
    for score in scores:
        if verbose:
            print("# Tuning hyper-parameters for {:s}".format(score))
            print()

        clf = GridSearchCV(SVC(), tuned_parameters, cv=cv,scoring=score)
        #clf = GridSearchCV(SVC(), tuned_parameters, cv=train_set, scoring=score)

        clf_l.append(clf)

        clf.fit(train_set[0], train_set[1])

        if verbose:
            b_params[score] = show_tuned_param_results(clf, verbose=show_param_test)

        if verbose:
            print("-------------        Detailed classification report:          --------------")
            print("-------------  The model is trained on the full training set  --------------")
            print()
            if validation_set is not None:
                c_reportsV[score] = generate_classification_report(clf, validation_set, target_names=t_names,
                                                                   set_type='Validation set', verbose=verbose)
            if test_set is not None:
                c_reportsT[score] = generate_classification_report(clf, test_set, target_names=t_names,
                                                                   set_type='Testing set', verbose=verbose)

    return clf, b_params, c_reportsV, c_reportsT, clf_l


# generates a test classification on the given data set using given classifier
def generate_classification_report(clf, data_set, target_names, set_type='validation data',
                                   verbose=True):
    print("--------  The scores are computed on the full {:s} ---------".format(set_type))
    print()
    print('mean accuracy: ', np.around(clf.score(data_set[0], data_set[1]), 2))
    y_true, y_pred = data_set[1], clf.predict(data_set[0])
    #print(classification_report(y_true, y_pred))
    c_report = classification_report(y_true, y_pred)
    rep_dic = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    if verbose:
        print(c_report)
        print()
    return rep_dic


# shows all tested parameter results
def show_tuned_param_results(clf, verbose=True):
    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    if verbose:
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
    return clf.best_params_


