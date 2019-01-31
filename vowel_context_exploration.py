import pandas as pd
from DataExplorerPD import *
from DataManipulation import *
from SVC_lib import grid_search_tester
from SVC_lib import z_normalize_svc

vowel_file = 'vowel-context.dt'

vowel_context = pd.read_csv(vowel_file, header=None, sep=' ')
#vowel_context.drop(13, axis=1, inplace=True)

data_explorer(vowel_context,type='Describe Info dtypes head')

# generate randomized training validation and test sets from data frame
train_set, validation_set, test_set, ks_ds = df_panda_splitter(vowel_context, seed=True, normalize=None)

# z normalize the cross validation sets
train_set, validation_set, test_set = z_normalize_svc(train_set, validation_set, test_set)

t_df = pd.DataFrame(train_set[1])
v_df = pd.DataFrame(validation_set[1])
s_df = pd.DataFrame(test_set[1])
print('Training data frame')
print(t_df.describe())
print()
print(t_df.isna())
print()
print(t_df.info(verbose=True))
print('--------------------------------------')
print('validation data frame')
print(v_df.describe())
print()
print(v_df.isna())
print()
print(v_df.info(verbose=True))
print('--------------------------------------')
print('Test data frame')
print(s_df.describe())
print()
print(s_df.isna())
print()
print(s_df.info(verbose=True))
print('--------------------------------------')

print('train set')
print(train_set[1])
print('val set')
print(validation_set[1])
print('test set')
print(test_set[1])
#quit(9)

# Set the parameters by cross-validation

# course grid search parameters
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [.1, 1e-2, 1e-3, 1e-4, 1e-5],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

l1 = [.1, .09, .08, .07, .06, .05, .04, .03, .02, 0.01, 0.009, 0.008, 0.007, 0.006, 0.005, 0.004, 0.003, 0.002,
       0.001]

tuned_parameters2 = [{'kernel': ['rbf'], 'gamma': np.around(np.linspace(.1,.01,20).tolist(), 2),
                     'C': list(range(1,10, 10))}]


tuned_parameters3 = [{'kernel': ['rbf'], 'gamma': np.around(np.linspace(.093, .094, 10).tolist(), 4),
                     'C': list(range(1,21, 1))}]

tuned_parameters4 = [{'kernel': ['rbf'], 'gamma': l1,
                     'C': list(range(1,21, 1))}]

tuned_parameters5 = [{'kernel': ['rbf'], 'gamma': np.around(np.linspace(.0930, .0931, 10).tolist(), 5),
                     'C': np.around(np.linspace(8.0, 9.0, 10).tolist(), 1)} ]

scores = ['precision_macro', 'recall_macro', 'accuracy']

scores1 = ['precision_macro', 'accuracy', 'balanced_accuracy']
scores2 = ['balanced_accuracy']

clf, b_params = grid_search_tester(tuned_parameters2, train_set, validation_set, test_set, cv=5, scores=scores,
                                   verbose=True)


for score in b_params:

    b_param = b_params[score]

    b_C = b_param['C']

    b_gamma = b_param['gamma']

    b_kernel = ['kernel']

    print('Score: {:s}'.format(score))
    print('best C:', b_C, 'best gama: ',b_gamma, 'best kernel: ', b_kernel)