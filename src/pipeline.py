import joblib
import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
from src.preprocessing import preprocess
from src.feature_selection import select_features
from src.imbalance import balance_data
from src.model import build_model


BASE_DIR = Path(__file__).resolve().parent.parent

# -----------------------------
# 1️⃣ Load Train and Test Separately
# -----------------------------
train = pd.read_csv(BASE_DIR / "data/ann-train.data", sep=r"\s+", header=None)
test = pd.read_csv(BASE_DIR / "data/ann-test.data", sep=r"\s+", header=None)

X_train = train.iloc[:, :-1]
y_train = train.iloc[:, -1]

X_test = test.iloc[:, :-1]
y_test = test.iloc[:, -1]
# -----------------------------
# 2️⃣ Cross Validation
# -----------------------------
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import numpy as np

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
macro_f1_scores = []

for train_idx, val_idx in skf.split(X_train, y_train):

    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    X_tr, imputer, scaler = preprocess(X_tr, fit=True)
    X_val = preprocess(X_val, fit=False, imputer=imputer, scaler=scaler)

    X_tr, selector = select_features(X_tr, y_tr, k=12)
    X_val = selector.transform(X_val)

    X_tr_bal, y_tr_bal = balance_data(X_tr, y_tr)

    model = build_model()
    model.fit(X_tr_bal, y_tr_bal)

    y_val_pred = model.predict(X_val)

    macro_f1_scores.append(
        f1_score(y_val, y_val_pred, average="macro")
    )

print("\nCross-Validation Macro F1:", np.mean(macro_f1_scores))

print("Original Train Distribution:")
print(y_train.value_counts())

print("\nOriginal Test Distribution:")
print(y_test.value_counts())

# -----------------------------
# 2️⃣ Preprocess (fit only on train)
# -----------------------------
X_train, imputer, scaler = preprocess(X_train, fit=True)
X_test = preprocess(X_test, fit=False, imputer=imputer, scaler=scaler)

# -----------------------------
# 3️⃣ Feature Selection (fit only on train)
# -----------------------------
X_train, selector = select_features(X_train, y_train, k=12)
X_test = selector.transform(X_test)

# -----------------------------
# 4️⃣ Apply SMOTE ONLY on train
# -----------------------------
X_train_bal, y_train_bal = balance_data(X_train, y_train)

print("\nAfter SMOTE Train Distribution:")
print(pd.Series(y_train_bal).value_counts())

# -----------------------------
# 5️⃣ Train Model
# -----------------------------
model = build_model()
model.fit(X_train_bal, y_train_bal)

# -----------------------------
# 6️⃣ Evaluate on REAL Test Set
# -----------------------------
y_pred = model.predict(X_test)

print("\nConfusion Matrix (Test Set):")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report (Test Set):")
print(classification_report(y_test, y_pred))

# -----------------------------
# 7️⃣ Save Artifacts
# -----------------------------
joblib.dump(imputer, BASE_DIR / "artifacts/imputer.pkl")
joblib.dump(scaler, BASE_DIR / "artifacts/scaler.pkl")
joblib.dump(selector, BASE_DIR / "artifacts/selector.pkl")
joblib.dump(model, BASE_DIR / "artifacts/model.pkl")

print("\nArtifacts saved successfully.")