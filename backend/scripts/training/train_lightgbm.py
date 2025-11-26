"""
LightGBM model training script.
"""
import os
import pickle
import time
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import randint, uniform
import sys
sys.path.insert(0, str(Path(__file__).parent))

from data_prep import load_and_prepare_data, get_cv_splits, prepare_features_target, get_train_test_split

try:
    import lightgbm as lgb
except ImportError:
    raise ImportError("LightGBM is not installed. Install it with: pip install lightgbm")


BASE_DIR = Path(__file__).resolve().parent.parent.parent  # This gives 'backend/'
DATA_PATH = BASE_DIR / 'data' / 'processed' / 'FEATURE_ENGINEERED_DATASET.csv'
MODELS_DIR = BASE_DIR / 'models'
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def train_lightgbm(evaluate_cv: bool = True, save_model: bool = True, use_hyperopt: bool = True):
    """
    Train LightGBM model with cross-validation evaluation.
    
    Args:
        evaluate_cv: Whether to perform cross-validation evaluation
        save_model: Whether to save the trained model
        use_hyperopt: Whether to use hyperparameter optimization
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
        best_params_list = []
        
        for fold, (train_index, test_index) in enumerate(tscv.split(X), start=1):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            if use_hyperopt:
                # Randomized search for hyperparameters
                lgb_model = lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1)
                inner_tscv = TimeSeriesSplit(n_splits=3)
                
                param_distributions = {
                    'n_estimators': randint(100, 500),
                    'max_depth': randint(3, 10),
                    'learning_rate': uniform(0.01, 0.2),
                    'num_leaves': randint(20, 100),
                    'subsample': uniform(0.6, 0.4),
                    'colsample_bytree': uniform(0.6, 0.4),
                    'min_child_samples': randint(10, 50)
                }
                
                random_search = RandomizedSearchCV(
                    estimator=lgb_model,
                    param_distributions=param_distributions,
                    n_iter=20,
                    cv=inner_tscv,
                    scoring='neg_mean_squared_error',
                    n_jobs=-1,
                    random_state=42,
                    verbose=0
                )
                
                random_search.fit(X_train, y_train)
                model = random_search.best_estimator_
                best_params_list.append(random_search.best_params_)
            else:
                # Use default parameters
                model = lgb.LGBMRegressor(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    num_leaves=31,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1
                )
                model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Measure inference time
            start_time = time.time()
            _ = model.predict(X_test[:100])  # Predict on 100 samples
            inference_time = (time.time() - start_time) / 100 * 1000  # ms per sample
            
            rmse_scores.append(rmse)
            mae_scores.append(mae)
            r2_scores.append(r2)
            inference_times.append(inference_time)
            
            print(f"Fold #{fold}: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}, Inference={inference_time:.2f}ms")
            if use_hyperopt:
                print(f"  Best params: {random_search.best_params_}")
        
        results['cv_rmse'] = np.mean(rmse_scores)
        results['cv_mae'] = np.mean(mae_scores)
        results['cv_r2'] = np.mean(r2_scores)
        results['cv_inference_time_ms'] = np.mean(inference_times)
        results['cv_std_rmse'] = np.std(rmse_scores)
        if use_hyperopt:
            results['best_params'] = best_params_list
        
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
    
    # Use best parameters from CV or defaults
    if use_hyperopt and 'best_params' in results:
        # Use most common best params (simplified - take first)
        best_params = results['best_params'][0]
        print(f"Using best parameters: {best_params}")
        model = lgb.LGBMRegressor(**best_params, random_state=42, n_jobs=-1, verbose=-1)
    else:
        model = lgb.LGBMRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            num_leaves=31,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
    
    model.fit(X_train_full, y_train_full)
    
    # Evaluate on test set
    y_pred_test = model.predict(X_test_full)
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
        model_path = MODELS_DIR / 'lightgbm.pkl'
        metadata_path = MODELS_DIR / 'lightgbm_metadata.pkl'
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        with open(metadata_path, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"  Model saved to: {model_path}")
        print(f"  Metadata saved to: {metadata_path}")
    
    return model, results


if __name__ == '__main__':
    train_lightgbm(evaluate_cv=True, save_model=True, use_hyperopt=True)

