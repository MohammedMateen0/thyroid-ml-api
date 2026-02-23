from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    StackingClassifier
)
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

def build_model():

    base_models = [

        # Tree Ensembles
        ("rf", RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            random_state=42
        )),

        ("brf", BalancedRandomForestClassifier(
            n_estimators=300,
            random_state=42
        )),

        ("et", ExtraTreesClassifier(
            n_estimators=400,
            random_state=42
        )),

        ("gb", GradientBoostingClassifier(
            n_estimators=300
        )),

        ("xgb", XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            eval_metric="mlogloss",
            random_state=42
        )),

        # Boosting variant
        ("ada", AdaBoostClassifier(
            n_estimators=300,
            random_state=42
        )),

        # Linear
        ("lr", LogisticRegression(
            max_iter=2000
        )),

        ("ridge", RidgeClassifier()),

        # Kernel
        ("svc", SVC(
            probability=True,
            kernel="rbf"
        )),

        # Distance
        ("knn", KNeighborsClassifier(
            n_neighbors=7
        )),

        # Probabilistic
        ("gnb", GaussianNB())
    ]

    meta_model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced"
    )

    stack = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_model,
        cv=5,
        n_jobs=-1,
        passthrough=False
    )

    return stack