import pandas as pd
from collections import OrderedDict


def load_wids_xy_data(path):
    """
    This function will format and condition the WIDS kaggle categorical data
    - will drop unnecessary columns
    - will fill NA's
    """

    print('loading data ...')
    df = pd.read_csv(path, low_memory=False)
    print('complete ...')

    print('formatting ...')

    # dropping sparsely populated columns
    drop_cols = ['LN2_RIndLngBEOth', 'LN2_WIndLngBEOth', 'train_id']
    drop_cols += [col for col in df.columns if 'REC' in col]
    drop_cols = [col for col in df.columns if 'OTHERS' in col]

    df.drop(columns=drop_cols, inplace=True)
    columns = [col for col in df.columns if col != 'is_female']
    y = df['is_female'].values
    X = df[columns].copy()

    print('imputing missing values ...')
    X.fillna(-1, inplace=True)
    print(X.shape, y.shape)
    return X, y


def get_mappers(inputX):
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
    for idx, col in enumerate(columns):
        if idx % 100 == 0:
            print(idx)
        X[col] = X[col].astype('category')
        mappers[col] = {labels: idx for idx, labels in enumerate(X[col].cat.categories)}

    print('calculating cardinality')
    categorical_stats = OrderedDict()
    for col in X.columns:
        categorical_stats[col] = len(X[col].cat.categories) + 1

    embedding_sizes = OrderedDict()
    for ky, vl in categorical_stats.items():
        embedding_sizes[ky] = (vl, min(50, (vl + 1) // 2))

    print('remapping columns to int')
    for col in columns:
        X[col] = X[col].map(mappers[col])

    emb_szs = list(embedding_sizes.values())
    print('complete')

    return X, mappers, categorical_stats, emb_szs
