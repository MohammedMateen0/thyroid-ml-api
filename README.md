# 🧠 ThyroAI – Machine Learning Based Thyroid Diagnosis System

An explainable machine learning web application for predicting thyroid disorders using clinical biomarkers and patient history.

Built with:

- FastAPI
- Scikit-Learn
- XGBoost (Stacked Ensemble)
- SHAP Explainability
- Docker
- Interactive Frontend Dashboard

---

## 📌 Problem Statement

Thyroid disorders such as Hypothyroidism and Hyperthyroidism are often underdiagnosed due to non-specific symptoms. This system uses machine learning to analyze:

- Blood biomarkers (TSH, T3, TT4, T4U, FTI)
- Clinical history flags
- Demographic data

to predict thyroid condition with confidence scoring and explainability.

---

## 🏗 System Architecture

```

User Input → FastAPI Backend →
Imputer → Scaler → Feature Selector →
Stacked Ensemble Model →
SHAP Explanation → JSON Response →
Interactive Dashboard

````

---

## 🧪 Machine Learning Pipeline

### Data Preprocessing
- Missing value imputation
- Standard scaling
- Feature selection

### Model
Stacked ensemble consisting of:
- Random Forest
- XGBoost
- Meta-classifier

### Output
- Final prediction
- Class probabilities
- Base model predictions
- SHAP feature impact analysis

---

## 📊 Features

- 🔍 Real-time prediction
- 📈 Probability visualization
- 🧩 Base model comparison
- 📉 SHAP explainability chart
- 🎯 Confidence donut visualization
- 🐳 Dockerized deployment
- ☁️ Cloud-ready architecture

---

## 🚀 Running Locally

### Using Python

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
````

Visit:

```
http://localhost:8000
```

---

### Using Docker

```bash
docker build -t thyroid-api .
docker run -p 8000:8000 thyroid-api
```

---

## ☁️ Deployment

This application is containerized and can be deployed on:

* Render
* AWS EC2
* Railway
* Docker-based cloud services

---

## 📁 Project Structure

```
app/
 ├── main.py
 └── templates/
      ├── home.html
      └── index.html

artifacts/
 ├── model.pkl
 ├── scaler.pkl
 ├── selector.pkl
 ├── imputer.pkl

Dockerfile
requirements.txt
README.md
```

---

## ⚠️ Disclaimer

This system is built for educational and research purposes only.
It does not replace professional medical diagnosis.

---

## 👨‍💻 Author

Mohammed Mateen
Machine Learning & Data Science Project
2026

````

