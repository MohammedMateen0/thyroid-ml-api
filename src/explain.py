import shap
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent

# ----------------------------
# Load Data (train only)
# ----------------------------
train = pd.read_csv(BASE_DIR / "data/ann-train.data", sep=r"\s+", header=None)
X = train.iloc[:, :-1]
y = train.iloc[:, -1]

# ----------------------------
# Load preprocessing artifacts
# ----------------------------
imputer = joblib.load(BASE_DIR / "artifacts/imputer.pkl")
scaler = joblib.load(BASE_DIR / "artifacts/scaler.pkl")
selector = joblib.load(BASE_DIR / "artifacts/selector.pkl")
model = joblib.load(BASE_DIR / "artifacts/model.pkl")

# ----------------------------
# Preprocess data
# ----------------------------
# ----------------------------
# Preprocess data
# ----------------------------
# Real feature names
feature_names = [
    "Age","Sex","On Thyroxine","Query on Thyroxine",
    "On Antithyroid Medication","Sick","Pregnant",
    "Thyroid Surgery","I131 Treatment","Query Hypothyroid",
    "Query Hyperthyroid","Lithium","Goitre","Tumor",
    "Hypopituitary","Psych","TSH","T3","TT4","T4U","FTI"
]

X_df = train.iloc[:, :-1]

# Apply preprocessing
X_processed = imputer.transform(X_df)
X_processed = scaler.transform(X_processed)

# Apply feature selection
X_selected = selector.transform(X_processed)

# Get selected indices
selected_indices = selector.get_support(indices=True)

# Map to real names
selected_feature_names = [feature_names[i] for i in selected_indices]

# Final DataFrame
X_final = pd.DataFrame(X_selected, columns=selected_feature_names)
# Extract XGBoost model from stack
# ----------------------------
xgb_model = None

for name, estimator in model.named_estimators_.items():
    if name == "xgb":
        xgb_model = estimator

# ----------------------------
# SHAP Explainer
# ----------------------------
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_final)

shap.summary_plot(shap_values[:, :, 1], X_final)
plt.show()
# ----------------------------
# Local explanation for one patient
# ----------------------------

# Choose a sample index (e.g., 0)
sample_index = 0

sample = X_final.iloc[sample_index:sample_index+1]

# Get SHAP values for Hyperthyroid class (index 1)
local_shap_values = shap_values[sample_index, :, 1]

# Force plot
shap.initjs()
force_plot = shap.force_plot(
    explainer.expected_value[1],
    local_shap_values,
    sample,
    matplotlib=True
)

plt.savefig(BASE_DIR / "artifacts/shap_force_sample0.png")
plt.show()