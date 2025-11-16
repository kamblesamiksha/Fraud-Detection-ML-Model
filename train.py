"""Train script for Fraud Detection model. Saves model.pkl and scaler.pkl in src/"""
import argparse
import joblib
import os
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report
from preprocess import load_data, feature_engineer, split_and_scale

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data-path', type=str, default='../data/transactions.csv')
    p.add_argument('--target', type=str, default='is_fraud')
    p.add_argument('--random-state', type=int, default=42)
    return p.parse_args()

def main():
    args = get_args()
    data = load_data(args.data_path)
    df = feature_engineer(data)
    X_train, X_test, y_train, y_test, scaler = split_and_scale(df, target_col=args.target, random_state=args.random_state)

    model = XGBClassifier(
        objective='binary:logistic',
        use_label_encoder=False,
        eval_metric='logloss',
        n_jobs=-1,
        random_state=args.random_state
    )

    param_dist = {
        'n_estimators': [100, 200, 400],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
    }

    search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=10, scoring='roc_auc', cv=3, verbose=2, random_state=args.random_state)
    search.fit(X_train, y_train)
    best = search.best_estimator_

    preds_proba = best.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, preds_proba)
    print(f'ROC-AUC on test: {auc:.4f}')
    print('Classification report:')
    print(classification_report(y_test, best.predict(X_test)))

    # Save model and scaler
    save_dir = os.path.dirname(__file__)
    joblib.dump(best, os.path.join(save_dir, 'model.pkl'))
    joblib.dump(scaler, os.path.join(save_dir, 'scaler.pkl'))
    print('Saved model.pkl and scaler.pkl to', save_dir)

if __name__ == '__main__':
    main()
