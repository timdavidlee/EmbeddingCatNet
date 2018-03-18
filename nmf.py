from sklearn.decomposition import NMF

def NMF_factorize(X, verbose=False):
    msk = X.isna()
    X = X.fillna(0)
    X_imputed = X.copy()
    
    # Initializing model
    nmf_model = NMF(n_components=5)
    nmf_model.fit(X_imputed.values)
    W = nmf_model.fit_transform(X_imputed.values)
    
    # iterate model
    while nmf_model.reconstruction_err_**2 > 10:
        W = nmf_model.fit_transform(X_imputed.values)
        X_imputed.values[~msk] = W.dot(nmf_model.components_)[~msk]
        if verbose:
            print(nmf_model.reconstruction_err_)
    
    return W, nmf_model.components_