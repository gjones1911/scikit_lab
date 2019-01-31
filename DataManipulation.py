import numpy as np
import pandas as pd

# ########################################################################################################
# ########################################################################################################
# ###########################################   cross validation   #######################################


def df_panda_splitter(data_frame, ptrn=.8, pval=.1, ptst=.1, normalize='z',
                      seed=False, seed_val=13, verbose=False, rand=True):

    return generate_data_sets_xy(data_frame.values, ptrn=ptrn, pval=pval, ptst=ptst, verbose=verbose, rand=rand,
                                 seed=seed, normalize=normalize, seed_val=seed_val)


def generate_data_sets_xy(data, ptrn=.80, pval=.10, ptst=.10, verbose=False, rand=False, seed=False, normalize=None,
                          seed_val=13, biased=False):

    data_shape = data.shape

    ks = data_shape[0]
    ds = data_shape[1]
    # print('data cols', ds)
    train_idx, val_idx, test_idx = split_data(len(data), p_train=ptrn, p_test=pval, p_val=ptst, verbose=verbose,
                                              rand=rand, seed=seed, seed_val=seed_val)

    train_set = np.array(get_cross_array(data, train_idx))
    tr_x = np.array(train_set[0:, 0:ds-1].tolist(), dtype=np.float64)
    if verbose:
        print('training attributes')
        print(tr_x)
    tr_r = np.array(train_set[0:, ds-1].tolist())
    if normalize is not None and normalize == 'z':
        mu = tr_x.mean(axis=0, dtype=np.float64)
        if verbose:
            print('mean: \n', mu)
        std = tr_x.std(axis=0, dtype=np.float64)
        if verbose:
            print('standard deviation: length = ', len(std))
            print(std)
        tr_x = (tr_x - mu)/std
    elif normalize is not None and normalize == 'n':
        mn = train_set.min(axis=0)
        mx = train_set.max(axis=0)
        tr_x = (tr_x - mn)/(mx - mn)

    if biased:
        tr_x = add_bias(tr_x)

    if verbose:
        print('Training observations: ', len(tr_x), 'Training attributes: ', len(tr_x[0]))
        print('Training classifications: ', len(tr_r))

    val_set = np.array(get_cross_array(data, val_idx))
    val_x = np.array(val_set[0:, 0:ds-1].tolist(), dtype=np.float64)
    val_r = np.array(val_set[0:, ds-1].tolist())

    if normalize is not None and normalize == 'z':
        val_x = (val_x-mu)/std
    elif normalize is not None and normalize == 'n':
        val_x = (val_x-mn)/(mx - mn)

    if biased:
        val_x = add_bias(val_x)
    if verbose:
        print('Validation observations: ', len(val_x), 'Validation attributes: ', len(val_x[0]))
        print('Validation classifications: ', len(val_r))

    test_set = np.array(get_cross_array(data, test_idx))
    ts_x = np.array(test_set[0:, 0:ds-1].tolist(), dtype=np.float64)
    ts_r = np.array(test_set[0:, ds-1].tolist())

    if normalize is not None and normalize == 'z':
        ts_x = (ts_x - mu)/std
    elif normalize is not None and normalize == 'n':
        ts_x = (ts_x - mn)/(mx - mn)

    if biased:
        ts_x = add_bias(ts_x)
    if verbose:
        print('Test observations: ',len(ts_x), 'Test attributes', len(ts_x[0]))
        print('Test classifications', len(ts_r))

    return list([tr_x, tr_r]), list([val_x, val_r]), list([ts_x, ts_r]), [ks, ds]


def split_data(data_size, p_train=.70, p_test=.30, p_val=.0, verbose=False, rand=True, seed=False, seed_val=None):

    trn_idx = list()
    tst_idx = list()
    val_idx = list()

    if rand:
        if seed:
            if seed_val is None:
                np.random.seed(data_size)
            else:
                np.random.seed(seed_val)
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
# ########################################################################################################
# ########################################################################################################
# #####################################  Data manipulation  ##############################################


def add_bias(data):
    d = data.tolist()
    ret_l = []
    for row in d:
        ret_l.append([1]+row)
    return np.array(ret_l, dtype=np.float64)


def adjust_data(file_name, new_file):

    f = open(file_name, 'r')
    nf = open(new_file, 'w')
    lines = f.readlines()

    for line in lines:
        dspc = line.find('  ')
        if dspc != -1:
            print(line)
            new_line = line[0:dspc] + line[dspc+1:]
            print(new_line)
            nf.write(new_line)
        else:
            nf.write(line)

    return


