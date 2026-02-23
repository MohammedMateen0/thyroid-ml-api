import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def preprocess(X, fit=True, imputer=None, scaler=None):
    if fit:
        imputer = SimpleImputer(strategy="median")
        X = imputer.fit_transform(X)

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        return X, imputer, scaler

    else:
        X = imputer.transform(X)
        X = scaler.transform(X)
        return X