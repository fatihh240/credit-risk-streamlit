# Credit Risk Prediction — Streamlit Demo

Interactive portfolio demo for **credit risk / loan default prediction** using a trained **LightGBM + scikit-learn Pipeline** and a tunable **decision threshold**.

## Features
- Form-based input for customer/loan attributes
- Risk probability (class=1) + final decision using a threshold slider
- Optional debug view of engineered features
- Simple “business impact (toy)” expected-loss panel

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
