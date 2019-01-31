import pandas as pd


def data_explorer(df, type='Describe'):
    if type.find('Describe') == 0:
        print('Data description')
        print(df.describe())
        print()
    if type.find('Info') != -1:
        print('Data Information')
        print(df.info())
        print()
    if type.find('dtypes') != -1:
        print('Data types:')
        print(df.dtypes)
        print()
    if type.find('head') != -1:
        print('First 5 rows:')
        print(df.head())
        print()

    return

