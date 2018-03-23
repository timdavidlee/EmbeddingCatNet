import pandas as pd
import numpy as np
from collections import OrderedDict


def load_wids_xy_data(path, target='is_female'):
    """
    This function will format and condition the WIDS kaggle categorical data
    - will drop unnecessary columns
    - will fill NA's
    """

    print('loading training data ...')
    df = pd.read_csv(path + '/train.csv', low_memory=False)
    print('loading test data ...')
    df_test = pd.read_csv(path + '/test.csv', low_memory=False)
    print('complete ...')

    print('formatting ...')

    # dropping sparsely populated columns
    drop_cols = ['LN2_RIndLngBEOth', 'LN2_WIndLngBEOth']
    drop_cols += [col for col in df.columns if 'REC' in col]
    drop_cols += [col for col in df.columns if 'OTHERS' in col]
    train_drop_cols = drop_cols + ['train_id']
    test_drop_cols = drop_cols + ['test_id']

    df.drop(columns=train_drop_cols, inplace=True)
    df_test.drop(columns=test_drop_cols, inplace=True)

    columns = [col for col in df.columns if col not in (['is_female'] + [target])]

    y = df[target].values
    X = df[columns].copy()
    X_test = df_test[columns].copy()

    print('imputing missing values ...')

    X.fillna(-1, inplace=True)
    X_test.fillna(-1, inplace=True)

    if target != 'is_female':
        y_test = df_test[target].values
        print(X.shape, y.shape, X_test.shape, y_test.shape)
        return X, y, X_test, y_test
    else:
        print(X.shape, y.shape, X_test.shape)
        return X, y, X_test


def get_mappers(inputX, cat_cols, emb_cols):
    """
    This function will take in a X pandas dataframe, turn all the data
    into categorical types, and re-map the values into integers. These
    mappings will be stored in 'mappers', this will also return
    embedding sizes used in the neural network:

    Example of embedding sizes:
        City
        -----
        - Los Angeles
        - New York
        - Houston
        - Portland
        - Atlanta

        This field would have an embedding size of (5, 2)
        - 5 unique values
        - 2 Embedding vector size (roughly half cardinality)


    X_mapped, mappers, categorical_stats, emb_szs = get_mappers(X)
    """
    X = inputX.copy()
    mappers = {}
    columns = X.columns

    print('converting to category ...')
    for idx, col in enumerate(cat_cols):
        if idx % 100 == 0:
            print(idx)
        X[col] = X[col].astype('category')
        mappers[col] = {labels: idx for idx, labels in enumerate(X[col].cat.categories)}

    print('calculating cardinality')
    categorical_stats = OrderedDict()
    for col in X.columns:
        categorical_stats[col] = len(X[col].cat.categories) + 1

    embedding_sizes = OrderedDict()
    for ky in emb_cols:
        vl = categorical_stats[ky]
        embedding_sizes[ky] = (vl, min(50, (vl + 1) // 2))

    print('remapping columns to int')
    for col in columns:
        X[col] = X[col].map(mappers[col])

    one_hot_cols = list(set(cat_cols).difference(set(emb_cols)))

    X = pd.get_dummies(X, columns=one_hot_cols).copy()

    emb_szs = embedding_sizes
    print('complete')

    idx2col = {idx: col for idx, col in enumerate(X.columns)}
    col2idx = {col: idx for idx, col in enumerate(X.columns)}

    return X, mappers, emb_szs, idx2col, col2idx


def get_trained_embeddings(mappers, model):
    keys = mappers.keys()
    emb_mtx = {}
    for field, emb in zip(keys, model.embs):
        emb_mtx[field] = emb.weight.data.numpy()
    return emb_mtx


def get_emb_df(X, emb_mtx, mappers):
    mini_dfs = []

    print('applying embeddings')
    for col in X.columns.values:
        idxs = X[col].map(mappers[col])

        # fill nones with global mean
        idxs[idxs.isna()] = max(idxs) + 1
        idxs = np.array(idxs, dtype=int)

        # get embedding matrix
        mtx = emb_mtx[col]

        # calculate global mean for missing values
        glb_mean = np.mean(mtx, axis=0)

        # add global mean to bottom of matrix
        mtx = np.concatenate([mtx, glb_mean.reshape(1, -1)], axis=0)

        # create dataframe
        jf = pd.DataFrame(mtx[idxs, :])
        jf.columns = [col + '_%d' % i for i in jf.columns]

        # append
        mini_dfs.append(jf)
    print('combining dfs')
    out_df = pd.concat(mini_dfs, axis=1)

    return out_df
