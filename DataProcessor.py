import numpy as np
from DisplayMethods import *
import ANN
import sys


# ################################################################################################################
# #####################################  Data processing cleaning  ###############################################
# ################################################################################################################
def generate_new_name(file_name, suffix):
    idx = file_name.index('.')
    new_name = file_name[0:idx] + suffix
    return new_name


# finds bad data points and returns a map
# keyed on the column and with a value of a list of the
# rows where the bad data is located
def find_col_bad_data(dataarray, badsig):
    retdic = {}
    for col in range(len(dataarray[0])):
            for row in range(len(dataarray)):
                if dataarray[row][col] == badsig:
                    if col in retdic:
                        retdic[col].append(row)
                    else:
                        rowlist = [row]
                        retdic[col] = rowlist
    return retdic


def convert_list(li, c_type='float'):

    if c_type == 'float':
        return list(map(float, li))
    elif c_type =='int':
        return list(map(int, li))
    else:
        print('unknown type no conversion!!!!')
        return li


# used to check a given line for a missing data signifier
# and returns the columns where they exist as a list
def check_line(line, sig):
    ret_l = {}
    for idx in range(line):
        if line[idx] == sig:
            ret_l.append(idx)
    return ret_l


def strip_string(line, chr):
    for cnt in range(line.count(chr)):
        line = line.strip(chr)
    return line


def rmv_col(li, rmv_li):
    for col in rmv_li:
        idx = li.index()
        li = li[:col] + li[col+1:]
    return li


def clean_split_line(line, strip_a=None, splitval=','):

    line
    if strip_a is None:
        strip_a = ['\n', ' ', '']

    for char in strip_a:
        line = strip_string(line, char)

    return line.split(splitval)


def process_data(file_name, strip_a=None, split_val=',', file_suffix='.dt', normalize=None, new_file=None):
    if strip_a is None:
        strip_a = ['\n', ' ', '']

    f = open(file_name, 'r')
    lines = f.readlines()

    indices = list(range(len(lines)))

    missing_dic = {}

    raw_data_attribs = []
    class_data = []

    for line, idx in zip(lines, indices):

        line = clean_split_line(line, strip_a, split_val)
        # missing_dic[idx] =
        raw_data_attribs.append(convert_list(line[:-1]))
        class_data.append(float(line[-1]))

    data_np1 = np.array(raw_data_attribs, dtype=np.float64)


    if normalize is not None and normalize == 'z_norm':
        data_np1 = (data_np1 - data_np1.mean(axis=0, dtype=np.float64))/data_np1.std(axis=0, dtype=np.float64)
    elif normalize is not None and normalize == 'normalize':
        data_np1 = (data_np1 - data_np1.min(axis=0))/(data_np1.max(axis=0) - data_np1.min(axis=0))

    print(data_np1[0])

    data_li = []

    for row,c in zip(data_np1.tolist(), class_data):
        data_li.append(row + list([c]))

    np_data = np.array(data_li, dtype=np.float64)

    if normalize is None:
        write_data_file(generate_new_name(file_name, '.dt'), np_data)
        return np_data


    #print(generate_new_name(file_name, '.dt'))

    write_data_file(generate_new_name(file_name, '.dt'), np_data)

    return np_data


def split_data(data_size, p_train=.70, p_test=.30, p_val=.0, verbose=False, rand=True, seed=False):

    trn_idx = list()
    tst_idx = list()
    val_idx = list()

    if rand:
        if seed:
            np.random.seed(data_size)
        r_c = np.random.choice(range(data_size), data_size, replace=False)
    else:
        r_c = list(range(data_size))

    train = int(np.around(data_size * p_train, 0))
    test = 0
    val = 0
    if p_val != 0:
        test = int(np.around(data_size * p_test, 0,))
        val = data_size - train - test
    else:
        test = data_size - train

    if verbose:
        print('train set size: ', train)
        print('test set size: ', test)
        print('val set size: ', val)

    for i in range(0, train):
        trn_idx.append(r_c[i])

    for i in range(train, train+test):
        tst_idx.append(r_c[i])

    for i in range(train+test, data_size):
        val_idx.append(r_c[i])

    if val == 0:
        return trn_idx, tst_idx
    else:
        return trn_idx, tst_idx, val_idx


def get_cross_array(data, indices):

    dl = data.tolist()

    ret_l = list()

    for idx in indices:
        ret_l.append(dl[idx])

    return ret_l


def generate_data_sets_xy(data, ptrn=.80, pval=.10, ptst=.10, verbose=False, rand=False, seed=False, normalize=None):
    '''
    bias_data = ANN.add_bias(data)
    '''

    data_shape = data.shape

    ks = data_shape[0]
    ds = data_shape[1]
    # print('data cols', ds)
    train_idx, val_idx, test_idx = split_data(len(data), p_train=ptrn, p_test=pval, p_val=ptst, verbose=verbose,
                                              rand=rand, seed=seed)

    train_set = np.array(get_cross_array(data, train_idx), dtype=np.float64)
    tr_x = np.array(train_set[0:, 0:ds-1].tolist(), dtype=np.float64)
    tr_r = train_set[0:, ds-1]
    if normalize is not None and normalize == 'z':
        mu = tr_x.mean(axis=0, dtype=np.float64)

        std = tr_x.std(axis=0, dtype=np.float64)
        tr_x = (tr_x - mu)/std
        tr_x = ANN.add_bias(tr_x)
    elif normalize is not None and normalize == 'n':
        mn = train_set.min(axis=0)
        mx = train_set.max(axis=0)
        tr_x = (tr_x - mn)/(mx - mn)
        tr_x = ANN.add_bias(tr_x)
    print('Training observations: ', len(tr_x), 'Training attributes: ', len(tr_x[0]))
    print('Training classifications: ', len(tr_r))

    val_set = np.array(get_cross_array(data, val_idx), dtype=np.float64)
    val_x = val_set[0:, 0:ds-1]
    val_r = val_set[0:, ds-1]

    if normalize is not None and normalize == 'z':
        val_x = (val_x-mu)/std
        val_x = ANN.add_bias(val_x)
    elif normalize is not None and normalize == 'n':
        val_x = (val_x-mn)/(mx - mn)
        val_x = ANN.add_bias(val_x)
    print('Validation observations: ', len(val_x), 'Validation attributes: ', len(val_x[0]))
    print('Validation classifications: ', len(val_r))

    test_set = np.array(get_cross_array(data, test_idx), dtype=np.float64)
    ts_x = test_set[0:, 0:ds-1]
    ts_r = test_set[0:, ds-1]

    if normalize is not None and normalize == 'z':
        ts_x = (ts_x - mu)/std
        ts_x = ANN.add_bias(ts_x)
    elif normalize is not None and normalize == 'n':
        ts_x = (ts_x - mn)/(mx - mn)
        ts_x = ANN.add_bias(ts_x)
    print('Test observations: ',len(ts_x), 'Test attributes', len(ts_x[0]))
    print('Test classifications', len(ts_r))

    return list([tr_x, tr_r]), list([val_x, val_r]), list([ts_x, ts_r]), [ks, ds]

# ################################################################################################################
# #################################  Data File writing and loading  ##############################################
# ################################################################################################################


def write_data_file(new_file, data):

    f = open(new_file, 'w')

    for r in range(len(data)):
        for c in range(len(data[r])):
            if c == len(data[r])-1:
                f.write(str(data[r][c]) + '\n')
            else:
                f.write(str(data[r][c]) + ',')
    f.close()
    return


def load_data_file(file_name, split_a=None, split_val=',', dtype='float'):

    f = open(file_name, 'r')

    lines = f.readlines()

    ret_array = []

    for line in lines:

        l = clean_split_line(line, split_a, split_val)

        if dtype == 'float':
            ret_array.append(list(map(float, clean_split_line(line, split_a, split_val))))
        elif dtype == 'int':
            ret_array.append(list(map(int, clean_split_line(line, split_a, split_val))))
        else:
            ret_array.append(clean_split_line(line, split_a, split_val))

    return np.array(ret_array, dtype=np.float64)

# ################################################################################################################
# ################################################################################################################
# ################################################################################################################


