# from __future__ import print_function
import numpy as np
import pandas as pd
import sklearn as skl
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC

from DataManipulation import *


file_name = 'ionosphere.data'

ionosphere_df = pd.read_csv(file_name, header=None)

print('original data frame:')
print(ionosphere_df)

ionosphere_df.drop(1, axis=1, inplace=True)
print('modified data frame:')
print(ionosphere_df)
print()
print()
print()
print()
print('The data as a numpy array')
print(ionosphere_df.values[:, 0:34])
print()
train_set, validation_set, test_set, ks_ds = df_panda_splitter(ionosphere_df, seed=True, verbose=True)

print('Training attributes:')
print(train_set[0])
print('Training classifications:')
print(train_set[1])

print()
print()
print('Validation set transformed:')
print(validation_set[0])
print('Validation classifications:')
print(validation_set[1])
print()
print()
print('Test set transformed:')
print(test_set[0])
print('Test classifications:')
print(test_set[1])



train_df = pd.DataFrame(train_set[0])

'''
print('descriptions')
print(train_df.info())
print('training data frame')
print(train_df.isna())
'''