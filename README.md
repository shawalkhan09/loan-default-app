# Loan Default Predictor

A Streamlit web app that predicts whether a loan applicant is likely to default, based on their financial and credit history. Built on the "Give Me Some Credit" dataset from Kaggle.

## Live Demo
🔗 [Try it here](https://loan-default-app-2tq6c3nfuti4k9gzt84ry9.streamlit.app/)

## How It Works

The app takes 10 applicant features as input — things like revolving credit utilization, age, payment history (30-59/60-89/90+ days late), debt ratio, income, open credit lines, real estate loans, and dependents — and predicts default risk along with a probability score.

**Input → Scale → Predict**
1. User fills in applicant details through the form
2. Inputs are scaled using a pre-fitted `StandardScaler`
3. A trained classification model outputs a prediction (default / no default) and probability

## Model

- Trained on the Give Me Some Credit dataset (Kaggle)
- Class imbalance handled with SMOTE
- Logistic Regression and Random Forest were compared; best model achieved **ROC-AUC ~0.85**
- Model and scaler are saved as `model.pkl` and `scaler.pkl` and loaded at runtime

## Tech Stack

- **Python**
- **Streamlit** - UI
- **scikit-learn** - model training
- **joblib** - model/scaler persistence
- **NumPy**

## Running Locally

```bash
git clone https://github.com/shawalkhan09/loan-default-app.git
cd loan-default-app
pip install -r requirements.txt
streamlit run app.py
```

Make sure `model.pkl` and `scaler.pkl` are in the same directory as `app.py`.

## Project Structure
loan-default-app/
├── app.py # Streamlit app
├── model.pkl # Trained classifier
├── scaler.pkl # Fitted StandardScaler
├── requirements.txt
└── README.md

## Disclaimer

This is a portfolio/learning project. Predictions are based on a public dataset and shouldn't be used for actual lending decisions.
