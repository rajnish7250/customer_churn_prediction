# Customer Churn Prediction (End-to-End ML Project)

## 📌 Overview
This project predicts whether a customer will churn based on behavioral and demographic data.

## 🚀 Features
- Data preprocessing & feature engineering
- Model training (Random Forest, Logistic Regression)
- Model evaluation (Accuracy, ROC-AUC)
- Flask API deployment

## 🛠 Tech Stack
Python, Pandas, NumPy, Scikit-learn, Flask

## ▶️ How to Run

1. Install dependencies:
pip install -r requirements.txt

2. Train model:
python src/train.py

3. Run API:
python app.py

## 📊 Sample API Request
POST /predict

{
    "features": [values_here]
}