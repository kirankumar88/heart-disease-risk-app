# Heart Disease Risk Predictor

A machine learning web application for predicting heart disease risk using a Random Forest classifier and Streamlit.
This project demonstrates an end-to-end workflow including data preprocessing, model optimization, evaluation, and deployment through an interactive clinical interface.

---

## Overview

This repository contains a Streamlit-based application that estimates cardiovascular risk from clinical parameters.
The application loads a tuned Random Forest model and provides probability-based predictions with severity tiers to assist risk interpretation.

The objective of this project is to showcase practical machine learning deployment within a healthcare analytics environment.

---

## Features

* Interactive clinical interface built with Streamlit
* Random Forest ensemble model
* Severity-tier based risk interpretation
* Probability-based prediction output
* Real-time clinical input handling
* Lightweight deployment-ready architecture

---

## Model Details

* Algorithm: Random Forest Classifier
* Hyperparameter Tuning: GridSearchCV
* Best Parameters:

  * n_estimators = 200
  * max_depth = 8
  * min_samples_leaf = 5
  * max_features = "sqrt"

### Model Performance

* Train Accuracy: **0.887**
* Test Accuracy: **0.786**
* ROC AUC: **0.815**
* Recall (Disease class): High sensitivity at lower thresholds

### Preprocessing

* One-Hot Encoding for categorical features
* Consistent feature alignment between training and deployment
* Encoder saved as `encoder.pkl`

---

## Input Variables

* Age
* Sex
* Chest Pain Type (cp)
* Resting Blood Pressure (trestbps)
* Cholesterol (chol)
* Fasting Blood Sugar (fbs)
* Rest ECG (restecg)
* Max Heart Rate (thalach)
* Exercise-Induced Angina (exang)
* Oldpeak
* Slope
* Thal

---

## 📁 Folder Structure

```id="hd_struct"
heart-disease-risk-app/
│
├── app.py            # Streamlit web application
├── best_rf.pkl       # Trained Random Forest model
├── encoder.pkl       # OneHotEncoder used during training
├── requirements.txt  # Python dependencies
├── README.md         # Project documentation
├── .gitignore        # Files excluded from version control
│
├── assets/
│     └── app_preview.png     # Application screenshot
│
└── notebooks/
      └── Heart_Disease_RF_Model.ipynb   # Model training & tuning
```

---

## Installation

Clone the repository:

```
git clone https://github.com/kirankumar88/heart-disease-risk-app
cd heart-disease-risk-app
```

Install dependencies:

```
pip install -r requirements.txt
```

Run the application:

```
python -m streamlit run app.py
```

---

## Application Preview

![App Screenshot](assets/app_preview.png)

---

## Risk Severity Tiers

* 🟢 Low Risk — Preventive lifestyle recommended
* 🟡 Moderate Risk — Lifestyle monitoring advised
* 🔴 High Risk — Clinical consultation recommended

---

## Disclaimer

This application is intended for educational and research purposes only.
It does not provide medical advice or clinical diagnosis.

---

## Author

**Kiran Kumar**
