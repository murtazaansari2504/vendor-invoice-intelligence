# Vendor Invoice Intelligence System

An end-to-end machine learning project using **Python**, **SQLite**, **Scikit-Learn**, and **Streamlit** to forecast vendor freight costs and flag high-risk invoices for manual audit and approval.

---

## Project Overview

Finance and procurement teams deal with high invoice volumes where manual review does not scale and abnormal freight or pricing patterns can cause cost leakage. This system provides two production-style modules:

1. **Freight Cost Prediction (Regression):** Estimates expected freight cost from invoice financial figures (`Dollars`) to support budgeting, landed-cost forecasting, and vendor negotiation.
2. **Invoice Risk Flagging (Classification):** Flags vendor invoices that warrant manual review due to unusual quantities, dollar amounts, freight charges, or delivery/payment patterns, allowing routine invoices to flow through faster.

---

## Data Source

Data is stored in a SQLite database (`data/inventory.db`) containing purchasing and vendor invoice records (such as purchase orders, receiving/invoice/payment dates, quantities, dollar amounts, and freight charges).

---

## Models & Pipeline Architecture

### 1. Freight Cost Prediction
* **Models evaluated:** Linear Regression, Decision Tree Regressor, and Random Forest Regressor using regression metrics (`MAE`, `RMSE`, `R²`).
* **Saved artifact:** Best model persisted as `models/predict_freight_model.pkl`.
* **Inference module:** `inference/predict_freight.py` loads the saved model and returns rounded freight cost predictions for new inputs.

### 2. Invoice Risk Flagging
* **Preprocessing & scaling:** Aggregates/prepares invoice features, splits train/test data, and fits a `StandardScaler` saved to `models/scaler.pkl`.
* **Model & tuning:** Trains a `RandomForestClassifier` with `GridSearchCV` using **F1 score** to balance precision and recall under class imbalance.
* **Saved artifacts:** Best classifier saved to `models/predict_flag_invoice.pkl` alongside `models/scaler.pkl`.
* **Inference module:** `inference/predict_invoice_flag.py` aligns input feature names and ordering, applies the fitted scaler, and outputs flag predictions (`flag_invoice` and `is_flagged`).

---

## Project Structure

```text
invoice generation/
├── data/
│   └── inventory.db
├── inference/
│   ├── predict_freight.py
│   └── predict_invoice_flag.py
├── invoice_falgging/          # or invoice_flagging/
│   ├── data_preprocessing.py
│   ├── modeling_evaluation.py
│   └── train.py
├── models/
│   ├── predict_freight_model.pkl
│   ├── predict_flag_invoice.pkl
│   └── scaler.pkl
├── app.py
└── README.md