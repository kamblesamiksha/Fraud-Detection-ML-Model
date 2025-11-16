"""Preprocessing utilities for Fraud Detection project."""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def load_data(path):
    df = pd.read_csv(path)
    return df

def feature_engineer(df):
    # Example feature engineering - adapt to your dataset
    df = df.copy()
    # Fill missing
    df.fillna(0, inplace=True)
    # Create example ratio feature if columns exist
    if 'transaction_amount' in df.columns and 'account_balance' in df.columns:
        df['amt_bal_ratio'] = df['transaction_amount'] / (df['account_balance'] + 1e-6)
    # Time based feature: hour of day if timestamp present
    if 'transaction_time' in df.columns:
        try:
            df['transaction_time'] = pd.to_datetime(df['transaction_time'])
            df['hour'] = df['transaction_time'].dt.hour
        except Exception:
            pass
    # One-hot encode small-cardinality categorical cols
    cat_cols = [c for c in df.select_dtypes(include=['object','category']).columns if df[c].nunique() < 50]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return df

def split_and_scale(df, target_col='is_fraud', test_size=0.2, random_state=42):
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # Apply SMOTE to training set only
    sm = SMOTE(random_state=random_state)
    X_res, y_res = sm.fit_resample(X_train_scaled, y_train)
    return X_res, X_test_scaled, y_res, y_test, scaler
