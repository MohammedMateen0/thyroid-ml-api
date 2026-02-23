import optuna
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from src.preprocessing import preprocess
from src.feature_selection import select_features
from src.imbalance import balance_data
from src.model import build_model

BASE_DIR = Path(__file__).resolve().parent.parent

# Load train data
train = pd.read_csv(BASE_DIR / "data/ann-train.data", sep=r"\s+", header=None)
X = train.iloc[:, :-1]
y = train.iloc[:, -1]

def objective(trial):

    # Suggest hyperparameters
    rf_n_estimators = trial.suggest_int("rf_n_estimators", 200, 600)
    rf_max_depth = trial.suggest_int("rf_max_depth", 5, 30)

    xgb_n_estimators = trial.suggest_int("xgb_n_estimators", 200, 600)
    xgb_lr = trial.suggest_float("xgb_lr", 0.01, 0.2)

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []

    for train_idx, val_idx in skf.split(X, y):

        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # preprocess
        X_tr, imputer, scaler = preprocess(X_tr, fit=True)
        X_val = preprocess(X_val, fit=False, imputer=imputer, scaler=scaler)

        # feature selection
        X_tr, selector = select_features(X_tr, y_tr, k=12)
        X_val = selector.transform(X_val)

        # SMOTE
        X_tr_bal, y_tr_bal = balance_data(X_tr, y_tr)

        # build model with trial parameters
        model = build_model(
            rf_n_estimators=rf_n_estimators,
            rf_max_depth=rf_max_depth,
            xgb_n_estimators=xgb_n_estimators,
            xgb_lr=xgb_lr
        )

        model.fit(X_tr_bal, y_tr_bal)

        y_pred = model.predict(X_val)
        scores.append(f1_score(y_val, y_pred, average="macro"))

    return sum(scores) / len(scores)


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)

print("Best Params:", study.best_params)
print("Best Macro F1:", study.best_value)