import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import (
    StandardScaler,
    label_binarize
)
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from imblearn.over_sampling import SMOTE

from src.model import build_model
from sklearn.metrics import classification_report

# =========================================================
# SETUP
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

print("Starting full professional evaluation...")

# =========================================================
# FEATURE & CLASS NAMES
# =========================================================

FEATURE_NAMES = [
    "Age","Sex","On Thyroxine","Query Thyroxine",
    "On Antithyroid Meds","Sick","Pregnant",
    "Thyroid Surgery","I131 Treatment","Query Hypothyroid",
    "Query Hyperthyroid","Lithium","Goitre","Tumor",
    "Hypopituitary","Psych","TSH","T3","TT4","T4U","FTI"
]

CLASS_NAMES = ["Hypothyroid","Hyperthyroid","Normal"]

# =========================================================
# LOAD DATA
# =========================================================

train = pd.read_csv(BASE_DIR / "data/ann-train.data", sep=r"\s+", header=None)
test = pd.read_csv(BASE_DIR / "data/ann-test.data", sep=r"\s+", header=None)

X_train = train.iloc[:, :-1]
y_train = train.iloc[:, -1]-1

X_test = test.iloc[:, :-1]
y_test = test.iloc[:, -1]-1
y_train = pd.Series(y_train).astype("category").cat.codes
y_test = pd.Series(y_test).astype("category").cat.codes

# =========================================================
# CLASS DISTRIBUTION BEFORE SMOTE
# =========================================================

plt.figure(figsize=(6,4))
y_train.value_counts().sort_index().plot(
    kind="bar",
    color=["#ef4444","#f59e0b","#10b981"]
)
plt.xticks(ticks=[0,1,2], labels=CLASS_NAMES, rotation=0)
plt.title("Class Distribution Before SMOTE")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "class_distribution_before_smote.png")
plt.close()

# =========================================================
# PREPROCESSING
# =========================================================

imputer = SimpleImputer(strategy="median")
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================================================
# FEATURE SELECTION (k=12)
# =========================================================

selector = SelectKBest(mutual_info_classif, k=12)
X_train_sel = selector.fit_transform(X_train, y_train)
X_test_sel = selector.transform(X_test)

selected_indices = selector.get_support(indices=True)
selected_feature_names = [FEATURE_NAMES[i] for i in selected_indices]

# =========================================================
# SMOTE
# =========================================================

smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_sel, y_train)

plt.figure(figsize=(6,4))
pd.Series(y_train_bal).value_counts().sort_index().plot(
    kind="bar",
    color=["#ef4444","#f59e0b","#10b981"]
)
plt.xticks(ticks=[0,1,2], labels=CLASS_NAMES, rotation=0)
plt.title("Class Distribution After SMOTE")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "class_distribution_after_smote.png")
plt.close()

# =========================================================
# CORRELATION HEATMAP
# =========================================================

plt.figure(figsize=(12,8))
sns.heatmap(
    pd.DataFrame(X_train_sel, columns=selected_feature_names).corr(),
    cmap="coolwarm",
    annot=False
)
plt.xticks(rotation=45)
plt.title("Correlation Heatmap (Selected Features)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "correlation_heatmap.png")
plt.close()

# =========================================================
# BUILD STACKING MODEL
# =========================================================

stack_model = build_model()
base_estimators = stack_model.estimators
all_models = {name: model for name, model in base_estimators}
all_models["Stacking"] = stack_model

# =========================================================
# TRAIN + EVALUATE ALL MODELS
# =========================================================

evaluation_results = []

print("Training and evaluating all models...\n")

for name, model in all_models.items():

    print(f"Training {name}")

    model.fit(X_train_bal, y_train_bal)
    y_pred = model.predict(X_test_sel)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted")
    rec = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    print("\n======================================")
    print(f"{name} MODEL PERFORMANCE")
    print("======================================")

    print(classification_report(
        y_test,
        y_pred,
        target_names=CLASS_NAMES
    ))
    print(f"\n{name} Evaluation")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1 Score  : {f1:.4f}")
        # ROC AUC
    macro_auc = None

    if hasattr(model, "predict_proba"):

        probs = model.predict_proba(X_test_sel)
        y_test_bin = label_binarize(y_test, classes=[0,1,2])

        auc_scores = []

        for i in range(3):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], probs[:, i])
            auc_score = auc(fpr, tpr)
            auc_scores.append(auc_score)

        macro_auc = np.mean(auc_scores)

    else:
        macro_auc = np.nan

    print(f"AUC Score : {macro_auc}")


    evaluation_results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "ROC_AUC": macro_auc
    })

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"confusion_matrix_{name}.png")
    plt.close()

metrics_df = pd.DataFrame(evaluation_results)
metrics_df.to_csv(OUTPUT_DIR / "all_model_metrics.csv", index=False)
metrics = ["Accuracy","Precision","Recall","F1","ROC_AUC"]
angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
angles = np.concatenate((angles,[angles[0]]))

fig = plt.figure(figsize=(8,8))
ax = plt.subplot(111, polar=True)

for _, row in metrics_df.iterrows():

    values = [
        row["Accuracy"],
        row["Precision"],
        row["Recall"],
        row["F1"],
        row["ROC_AUC"]
    ]

    values += values[:1]

    ax.plot(angles, values, linewidth=1.5, label=row["Model"])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics)

plt.title("Radar Comparison of All Models")
plt.legend(loc="upper right", bbox_to_anchor=(1.3,1.1))

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "radar_all_models.png")
plt.close()
plt.figure(figsize=(14,6))
sorted_df = metrics_df.sort_values("ROC_AUC", ascending=False)

sns.barplot(
    data=sorted_df,
    x="Model",
    y="ROC_AUC",
    hue="Model",
    palette="coolwarm",
    legend=False
)

plt.xticks(rotation=45)
plt.title("ROC-AUC Comparison Across All Models")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "roc_auc_all_models.png")
plt.close()
# =========================================================
# LEARNING CURVE COMPARISON (ALL MODELS)
# =========================================================

from sklearn.model_selection import learning_curve

train_sizes = np.linspace(0.1, 1.0, 5)

plt.figure(figsize=(12,7))

for name, model in all_models.items():

    train_sizes_abs, train_scores, val_scores = learning_curve(
        model,
        X_train_sel,
        y_train,
        cv=5,
        scoring="f1_macro",
        train_sizes=train_sizes,
        n_jobs=-1
    )

    val_mean = np.mean(val_scores, axis=1)

    plt.plot(
        train_sizes_abs,
        val_mean,
        marker="o",
        label=name
    )

plt.xlabel("Training Samples")
plt.ylabel("Validation F1 Score")
plt.title("Learning Curve Comparison Across Models")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "learning_curve_all_models.png")
plt.close()

# =========================================================
# ACCURACY COMPARISON
# =========================================================

plt.figure(figsize=(14,6))
sorted_df = metrics_df.sort_values("Accuracy", ascending=False)

sns.barplot(
    data=sorted_df,
    x="Model",
    y="Accuracy",
    hue="Model",
    palette="viridis",
    legend=False
)
plt.xticks(rotation=45)
plt.title("Accuracy Comparison Across All Models")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "accuracy_all_models.png")
plt.close()

# =========================================================
# F1 SCORE COMPARISON
# =========================================================

plt.figure(figsize=(14,6))
sorted_df = metrics_df.sort_values("F1", ascending=False)

sns.barplot(
    data=sorted_df,
    x="Model",
    y="F1",
    hue="Model",
    palette="magma",
    legend=False
)
plt.xticks(rotation=45)
plt.title("F1 Score Comparison Across All Models")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "f1_all_models.png")
plt.close()

# =========================================================
# CROSS VALIDATION COMPARISON
# =========================================================

cv_results = []
model_names = []

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in all_models.items():

    fold_scores = []

    for train_idx, val_idx in skf.split(X_train_sel, y_train):

        X_tr, X_val = X_train_sel[train_idx], X_train_sel[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

# ensure continuous labels
        y_tr = pd.Series(y_tr).astype("category").cat.codes
        y_val = pd.Series(y_val).astype("category").cat.codes

        X_tr_bal, y_tr_bal = smote.fit_resample(X_tr, y_tr)

        model.fit(X_tr_bal, y_tr_bal)
        y_val_pred = model.predict(X_val)

        score = f1_score(y_val, y_val_pred, average="macro")
        fold_scores.append(score)

    cv_results.append(fold_scores)
    model_names.append(name)

plt.figure(figsize=(14,6))
plt.boxplot(cv_results, tick_labels=model_names)
plt.xticks(rotation=45)
plt.title("Cross Validation Macro F1 Comparison")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "cv_comparison_all_models.png")
plt.close()

# =========================================================
# ROC CURVE (STACKING MULTICLASS)
# =========================================================

stack_model.fit(X_train_bal, y_train_bal)
probs = stack_model.predict_proba(X_test_sel)
y_test_bin = label_binarize(y_test, classes=[0,1,2])

plt.figure(figsize=(8,6))

for i in range(3):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], probs[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{CLASS_NAMES[i]} (AUC={roc_auc:.2f})")

plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Multiclass ROC Curve - Stacking")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "roc_multiclass_stacking.png")
plt.close()

# =========================================================
# XGBOOST FEATURE IMPORTANCE (NAMED)
# =========================================================

if "xgb" in dict(base_estimators):
    xgb_model = dict(base_estimators)["xgb"]
    xgb_model.fit(X_train_bal, y_train_bal)

    importances = xgb_model.feature_importances_
    sorted_idx = np.argsort(importances)

    plt.figure(figsize=(8,6))
    plt.barh(
        np.array(selected_feature_names)[sorted_idx],
        importances[sorted_idx],
        color="#6366f1"
    )
    plt.title("XGBoost Feature Importance")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "xgb_feature_importance_named.png")
    plt.close()

    # SHAP
# =========================================================
# SHAP SUMMARY (FIXED MULTICLASS)
# =========================================================

explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test_sel)

# For modern SHAP versions:
if len(shap_values.shape) == 3:
    # shape: (samples, features, classes)
    shap_class = shap_values[:, :, 1]  # Hyperthyroid class
else:
    # older SHAP format
    shap_class = shap_values[1]

shap.summary_plot(
    shap_class,
    X_test_sel,
    feature_names=selected_feature_names,
    show=False
)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "shap_summary_named.png")
plt.close()

print("\nAll professional evaluation plots saved in /outputs/")