"""
CatBoost model training script.
CatBoost often outperforms XGBoost/LightGBM with minimal tuning,
especially when data has categorical features.
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path
import sys
import catboost as cb
from scipy.stats import randint, uniform
import time

sys.path.insert(0, str(Path(__file__).parent))
from data_prep import load_and_prepare_data, prepare_features_target, get_train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent.parent #backend/

DATA_PATH = BASE_DIR/'data'/'processed'/'FEATURE_ENGINEERED_DATASET.csv'

MODEL_DIR = BASE_DIR/'models'
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def train_catboost(use_hyperparameter):
    """Train CatBoost model and evaluate it"""
    print("Loading data...")
    data = load_and_prepare_data(str(DATA_PATH))
    X, y = prepare_features_target(data)

    print(X.shape)
    print(list(X.columns[:5]))
#use last 1800 hours for testing, with 168 hours gap (1 week)
    train_data, test_data = get_train_test_split(data, test_size=1800, gap=168)
    
    X_train, y_train = prepare_features_target(train_data)
    X_test, y_test = prepare_features_target(test_data)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test data: {len(X_test)}")
    if use_hyperparameter:
        param_grid = {
            'depth': [8, 10, 12, 14] , # Go deeper
            'iterations': [300, 400, 500],
            'learning_rate': [0.05, 0.1, 0.15],
            'l2_leaf_reg': [1,3,5]
        }
        model = cb.CatBoostRegressor(verbose=0)
        grid_search = GridSearchCV(
        model, param_grid, cv=3, scoring='neg_mean_squared_error',
        verbose=100)
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
        print(f"Best params: {grid_search.best_params_}")
    else:
        # Use default parameters
        model = cb.CatBoostRegressor(
            iterations=5000,
            depth=6,
            learning_rate=0.03,
            l2_leaf_reg=1,
            verbose=100
        )
        model.fit(X_train, y_train)

    
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"RMSE: {rmse:.2f}")
    
    # Save the model
    model_path = MODEL_DIR / 'catboost.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"✅ Model saved to {model_path}")


if __name__ == '__main__':
    train_catboost(use_hyperparameter=False)