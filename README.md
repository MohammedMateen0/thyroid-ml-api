# 🧬 ThyroAI – Explainable Machine Learning Based Thyroid Diagnosis System

A complete end-to-end Machine Learning system for automated thyroid disorder classification using ensemble learning and explainable AI (SHAP).

This project includes:

- Data preprocessing pipeline
- Feature selection
- Class imbalance handling (SMOTE)
- 11 base models
- Stacking ensemble
- Model evaluation & visualization
- Explainability using SHAP
- FastAPI deployment
- Docker support

---

## 📌 Problem Statement

Thyroid disorders such as Hyperthyroidism and Hypothyroidism are common endocrine conditions. Early detection is critical for effective treatment.

This project builds a robust ensemble machine learning system that:

- Classifies patients into:
  - Normal
  - Hyperthyroid
  - Hypothyroid
- Uses clinical + biochemical features
- Provides explainable predictions
- Compares multiple ML models
- Produces publication-level evaluation graphs

---

## 📊 Dataset

Source: UCI Thyroid Disease Dataset (ANN dataset)

- Training file: `ann-train.data`
- Test file: `ann-test.data`
- Total features: 21
- Selected features (after feature selection): 12

### Features Used

- Age
- Sex
- On Thyroxine
- Query Thyroxine
- On Antithyroid Medication
- Sick
- Pregnant
- Thyroid Surgery
- I131 Treatment
- Query Hypothyroid
- Query Hyperthyroid
- Lithium
- Goitre
- Tumor
- Hypopituitary
- Psych
- TSH
- T3
- TT4
- T4U
- FTI

---

## ⚙️ Project Architecture

```
Raw Data
   ↓
Imputation (Median)
   ↓
Standard Scaling
   ↓
Feature Selection (Mutual Information - Top 12)
   ↓
SMOTE (Class Balancing)
   ↓
Base Models (11)
   ↓
Stacking Ensemble
   ↓
Evaluation & Explainability
   ↓
FastAPI Deployment
```

---

## 🤖 Models Used

### Base Models

- Random Forest
- Balanced Random Forest
- Extra Trees
- Gradient Boosting
- XGBoost
- AdaBoost
- Logistic Regression
- Ridge Classifier
- Support Vector Machine (RBF)
- K-Nearest Neighbors
- Gaussian Naive Bayes

### Meta Model (Stacking)

- Logistic Regression (balanced)

---

## 📈 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix (per model)
- Multiclass ROC Curve
- Cross Validation (5-Fold Macro F1)
- Feature Importance (XGBoost)
- SHAP Global Feature Importance

---

## 📊 Generated Evaluation Outputs

All plots are saved inside:

```
/outputs/
```

### Generated Graphs:

- Class Distribution (Before SMOTE)
![alt text](class_distribution_before_smote.png)
- Class Distribution (After SMOTE)
![alt text](class_distribution_after_smote.png)
- Correlation Heatmap
![alt text](correlation_heatmap.png)
- Confusion Matrix (for each model)
- Accuracy Comparison (All Models)
![alt text](confusion_matrix_ada.png)
![alt text](confusion_matrix_brf.png)
![alt text](confusion_matrix_et.png)
![alt text](confusion_matrix_gb.png) 
![alt text](confusion_matrix_gnb.png) 
![alt text](confusion_matrix_knn.png) 
![alt text](confusion_matrix_rf.png) 
![alt text](confusion_matrix_ridge.png) 
![alt text](confusion_matrix_Stacking.png) 
![alt text](confusion_matrix_svc.png)
![alt text](confusion_matrix_lr.png) 
![alt text](confusion_matrix_xgb.png)
- F1 Score Comparison
![alt text](f1_all_models.png)
- Cross-Validation Comparison
![alt text](cv_comparison_all_models.png)
- Multiclass ROC Curve
![alt text](roc_multiclass_stacking.png)
- XGBoost Feature Importance
![alt text](xgb_feature_importance_named.png)
- SHAP Summary Plot
![alt text](shap_summary_named.png)

---

## 🧠 Explainability (SHAP)

SHAP is used to:

- Identify most influential biomarkers
- Understand class-specific predictions
- Provide transparent AI decisions

Example interpretation:

> TSH and FTI show highest contribution in differentiating Hyperthyroid cases.

---

## 🚀 Running the Evaluation

Activate virtual environment:

```bash
venv\Scripts\activate
```

Run full evaluation:

```bash
python paper_evaluation.py
```

All plots will be saved inside `/outputs/`.

---

## 🌐 FastAPI Deployment

Run locally:

```bash
uvicorn app.main:app --reload
```

Open:

```
http://127.0.0.1:8000
```

Endpoints:

- `/predict`
- `/predict-batch`
- `/health`

---

## 🐳 Docker Support

Build image:

```bash
docker build -t thyroid-api .
```

Run container:

```bash
docker run -p 8000:8000 thyroid-api
```

---

## 📦 Project Structure

```
ML42/
│
├── app/                     # FastAPI app
│   ├── main.py
│   └── templates/
│
├── data/
│   ├── ann-train.data
│   └── ann-test.data
│
├── src/
│   ├── model.py
│   ├── preprocessing.py
│   ├── feature_selection.py
│   └── ...
│
├── outputs/                 # Evaluation plots
├── artifacts/               # Saved models
├── paper_evaluation.py
├── requirements.txt
└── Dockerfile
```

---

## 🏆 Key Contributions

- Large ensemble of 11 models
- Stacking meta-learner
- Feature selection for dimensionality reduction
- SMOTE for class balancing
- Full comparative evaluation
- Explainable AI integration
- Production-ready API
- Docker deployment ready

---

## 📚 Technologies Used

- Python 3.11
- Scikit-learn
- XGBoost
- Imbalanced-learn
- SHAP
- Matplotlib
- Seaborn
- FastAPI
- Docker

---

## 📌 Academic Relevance

This project demonstrates:

- Advanced ensemble learning
- Model comparison methodology
- Multiclass evaluation
- Explainable AI in healthcare
- Real-world ML deployment pipeline

Suitable for:

- Final Year Major Project
- Machine Learning Coursework
- Research Demonstration
- Clinical AI Prototype

---

## ⚠️ Disclaimer

This system is built for educational and research purposes only.  
It is not a replacement for professional medical diagnosis.

---

## 👨‍💻 Author

Mohammed Mateen  
Machine Learning & Data Science Enthusiast  
Hyderabad, India