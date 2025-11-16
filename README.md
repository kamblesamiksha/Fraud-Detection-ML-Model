# Fraud Detection ML Model

## Overview
This repository contains code to train and deploy a fraud detection model using XGBoost with SMOTE balancing.
It includes preprocessing, training, and a FastAPI-based inference service.

## Structure
- data/transactions.csv (place your dataset here)
- src/preprocess.py
- src/train.py
- src/predict_api.py
- requirements.txt

## Quick start
1. Create a virtual environment: `python -m venv venv && source venv/bin/activate`
2. Install dependencies: `pip install -r requirements.txt`
3. Place your dataset at `data/transactions.csv` (see notes below for expected format).
4. Train the model: `python src/train.py --data-path data/transactions.csv`
5. Run the API: `uvicorn src.predict_api:app --reload --port 8000`

## Data expectations
The training script expects a CSV with a binary target column named `is_fraud` (1 for fraud, 0 for legitimate).
Include transactional columns like `transaction_amount`, `transaction_time`, `merchant_id`, `user_id`, etc.
Adjust feature engineering in `src/preprocess.py` to match your schema.

## Notes
- The training script will save `model.pkl` and `scaler.pkl` into `src/` after training.
- For production, replace joblib/pickle model save with a model registry (MLflow) and secure secrets for API keys.
