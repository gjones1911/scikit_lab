import numpy as np
import pandas as pd
import sklearn as skl



file_name = 'ionosphere.data'

col_num = list(range(34))
col_num.append('g/b')

#df = pd.DataFrame(pd.read_csv(file_name), columns=col_num)


df = pd.read_csv(file_name,header=None)

print('data types')
print(df.dtypes)

print('------------------------is na?')
print(df.isna)

print('head 2')
print(df.head(2))

print('description:')
print(df.describe())
d_stats = df.describe()

print()
print('mean of df:')
print(df.mean())
print()
print('standard deviation:')
print(d_stats.std(axis=0))
print()
print('info:')
print(df.info())
print()
print('columns:')
print(df.columns)
print()
print('classes:')
signal_classes = df[[34]]
print(signal_classes)
print()
print('head 4')
print(df.head(4))
print()
print('location slicing: 2, 4')
print(list(df.iloc[2]))

