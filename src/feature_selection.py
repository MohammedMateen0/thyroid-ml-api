from sklearn.feature_selection import SelectKBest, mutual_info_classif

def select_features(X, y, k=12):
    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    X_new = selector.fit_transform(X, y)
    return X_new, selector