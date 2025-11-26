"""
Linear Regression model training script.
Extracted from linear_regression_v2.ipynb and standardized.
"""
import os
import pickle
import time
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from data_prep import load_and_prepare_data, get_cv_splits, prepare_features_target, get_train_test_split


BASE_DIR = Path(__file__).resolve().parent.parent.parent  # This gives 'backend/'
DATA_PATH = BASE_DIR / 'data' / 'processed' / 'FEATURE_ENGINEERED_DATASET.csv'
MODELS_DIR = BASE_DIR / 'models'
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def train_linear_regression(evaluate_cv: bool = True, save_model: bool = True):
    """
    Train Linear Regression model with cross-validation evaluation.
    
    Args:
        evaluate_cv: Whether to perform cross-validation evaluation
        save_model: Whether to save the trained model
    """
    print("Loading and preparing data...")
    data = load_and_prepare_data(str(DATA_PATH))
    X, y = prepare_features_target(data)
    
    results = {}
    
    if evaluate_cv:
        print("\nPerforming cross-validation...")
        tscv = get_cv_splits(X, n_splits=4, test_size=1800, gap=168)
        
        rmse_scores = []
        mae_scores = []
        r2_scores = []
        inference_times = []
        
        for fold, (train_index, test_index) in enumerate(tscv.split(X), start=1):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model = LinearRegression()
            model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test_scaled)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Measure inference time
            start_time = time.time()
            _ = model.predict(X_test_scaled[:100])  # Predict on 100 samples
            inference_time = (time.time() - start_time) / 100 * 1000  # ms per sample
            
            rmse_scores.append(rmse)
            mae_scores.append(mae)
            r2_scores.append(r2)
            inference_times.append(inference_time)
            
            print(f"Fold #{fold}: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}, Inference={inference_time:.2f}ms")
        
        results['cv_rmse'] = np.mean(rmse_scores)
        results['cv_mae'] = np.mean(mae_scores)
        results['cv_r2'] = np.mean(r2_scores)
        results['cv_inference_time_ms'] = np.mean(inference_times)
        results['cv_std_rmse'] = np.std(rmse_scores)
        
        print(f"\nCross-Validation Results:")
        print(f"  Average RMSE: {results['cv_rmse']:.4f} ± {results['cv_std_rmse']:.4f}")
        print(f"  Average MAE: {results['cv_mae']:.4f}")
        print(f"  Average R²: {results['cv_r2']:.4f}")
        print(f"  Average Inference Time: {results['cv_inference_time_ms']:.2f}ms")
    
    # Train on full dataset
    print("\nTraining on full dataset...")
    train_data, test_data = get_train_test_split(data, test_size=1800, gap=168)
    X_train_full, y_train_full = prepare_features_target(train_data)
    X_test_full, y_test_full = prepare_features_target(test_data)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_full)
    X_test_scaled = scaler.transform(X_test_full)
    
    # Train final model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train_full)
    
    # Evaluate on test set
    y_pred_test = model.predict(X_test_scaled)
    test_rmse = np.sqrt(mean_squared_error(y_test_full, y_pred_test))
    test_mae = mean_absolute_error(y_test_full, y_pred_test)
    test_r2 = r2_score(y_test_full, y_pred_test)
    
    results['test_rmse'] = test_rmse
    results['test_mae'] = test_mae
    results['test_r2'] = test_r2
    results['feature_names'] = list(X.columns)
    
    print(f"\nTest Set Results:")
    print(f"  RMSE: {test_rmse:.4f}")
    print(f"  MAE: {test_mae:.4f}")
    print(f"  R²: {test_r2:.4f}")
    
    if save_model:
        print("\nSaving model...")
        model_path = MODELS_DIR / 'linear_regression.pkl'
        scaler_path = MODELS_DIR / 'linear_regression_scaler.pkl'
        metadata_path = MODELS_DIR / 'linear_regression_metadata.pkl'
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        with open(metadata_path, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"  Model saved to: {model_path}")
        print(f"  Scaler saved to: {scaler_path}")
        print(f"  Metadata saved to: {metadata_path}")
    
    return model, scaler, results


if __name__ == '__main__':
    train_linear_regression(evaluate_cv=True, save_model=True)

