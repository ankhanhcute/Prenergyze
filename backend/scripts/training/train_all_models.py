"""
Train all models and compare their performance.
This script trains Linear Regression, Random Forest, XGBoost, LightGBM, and LSTM,
then saves a comparison report.
"""
import json
import pickle
from pathlib import Path
from datetime import datetime

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from linear_regression import train_linear_regression
from random_forest import train_random_forest
from train_xgboost import train_xgboost
from train_lightgbm import train_lightgbm
from lstm import train_lstm


BASE_DIR = Path(__file__).resolve().parent.parent.parent  # This gives 'backend/'
MODELS_DIR = BASE_DIR / 'models'
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def train_all_models():
    """Train all models and create comparison report."""
    print("=" * 80)
    print("Training All Models for Ensemble Selection")
    print("=" * 80)
    
    results = {}
    
    # Train Linear Regression
    print("\n" + "=" * 80)
    print("1. Training Linear Regression")
    print("=" * 80)
    try:
        _, _, lr_results = train_linear_regression(evaluate_cv=True, save_model=True)
        results['linear_regression'] = {
            'cv_rmse': lr_results.get('cv_rmse'),
            'cv_mae': lr_results.get('cv_mae'),
            'cv_r2': lr_results.get('cv_r2'),
            'cv_inference_time_ms': lr_results.get('cv_inference_time_ms'),
            'test_rmse': lr_results.get('test_rmse'),
            'status': 'success'
        }
    except Exception as e:
        print(f"Error training Linear Regression: {e}")
        results['linear_regression'] = {'status': 'failed', 'error': str(e)}
    
    # Train Random Forest
    print("\n" + "=" * 80)
    print("2. Training Random Forest")
    print("=" * 80)
    try:
        _, rf_results = train_random_forest(evaluate_cv=True, save_model=True, use_grid_search=True)
        results['random_forest'] = {
            'cv_rmse': rf_results.get('cv_rmse'),
            'cv_mae': rf_results.get('cv_mae'),
            'cv_r2': rf_results.get('cv_r2'),
            'cv_inference_time_ms': rf_results.get('cv_inference_time_ms'),
            'test_rmse': rf_results.get('test_rmse'),
            'status': 'success'
        }
    except Exception as e:
        print(f"Error training Random Forest: {e}")
        results['random_forest'] = {'status': 'failed', 'error': str(e)}
    
    # Train XGBoost
    print("\n" + "=" * 80)
    print("3. Training XGBoost")
    print("=" * 80)
    try:
        _, xgb_results = train_xgboost(evaluate_cv=True, save_model=True, use_hyperopt=True)
        results['xgboost'] = {
            'cv_rmse': xgb_results.get('cv_rmse'),
            'cv_mae': xgb_results.get('cv_mae'),
            'cv_r2': xgb_results.get('cv_r2'),
            'cv_inference_time_ms': xgb_results.get('cv_inference_time_ms'),
            'test_rmse': xgb_results.get('test_rmse'),
            'status': 'success'
        }
    except Exception as e:
        print(f"Error training XGBoost: {e}")
        results['xgboost'] = {'status': 'failed', 'error': str(e)}
    
    # Train LightGBM
    print("\n" + "=" * 80)
    print("4. Training LightGBM")
    print("=" * 80)
    try:
        _, lgb_results = train_lightgbm(evaluate_cv=True, save_model=True, use_hyperopt=True)
        results['lightgbm'] = {
            'cv_rmse': lgb_results.get('cv_rmse'),
            'cv_mae': lgb_results.get('cv_mae'),
            'cv_r2': lgb_results.get('cv_r2'),
            'cv_inference_time_ms': lgb_results.get('cv_inference_time_ms'),
            'test_rmse': lgb_results.get('test_rmse'),
            'status': 'success'
        }
    except Exception as e:
        print(f"Error training LightGBM: {e}")
        results['lightgbm'] = {'status': 'failed', 'error': str(e)}
    
    # Train LSTM
    print("\n" + "=" * 80)
    print("5. Training LSTM")
    print("=" * 80)
    try:
        _, _, _, lstm_results = train_lstm(
            evaluate_cv=True,
            save_model=True,
            sequence_length=24,
            hidden_size=64,
            num_layers=2,
            dropout=0.2,
            epochs=50,
            patience=10
        )
        results['lstm'] = {
            'cv_rmse': lstm_results.get('cv_rmse'),
            'cv_mae': lstm_results.get('cv_mae'),
            'cv_r2': lstm_results.get('cv_r2'),
            'cv_inference_time_ms': lstm_results.get('cv_inference_time_ms'),
            'test_rmse': lstm_results.get('test_rmse'),
            'status': 'success'
        }
    except Exception as e:
        print(f"Error training LSTM: {e}")
        results['lstm'] = {'status': 'failed', 'error': str(e)}
    
    # Create comparison report
    print("\n" + "=" * 80)
    print("Model Comparison Summary")
    print("=" * 80)
    
    comparison = {
        'timestamp': datetime.now().isoformat(),
        'models': results,
        'ranking': []
    }
    
    # Rank models by CV RMSE
    successful_models = [
        (name, data) for name, data in results.items()
        if data.get('status') == 'success' and data.get('cv_rmse') is not None
    ]
    successful_models.sort(key=lambda x: x[1]['cv_rmse'])
    
    print("\nModel Rankings (by CV RMSE):")
    for rank, (name, data) in enumerate(successful_models, 1):
        print(f"{rank}. {name}:")
        print(f"   CV RMSE: {data['cv_rmse']:.4f}")
        print(f"   CV MAE: {data['cv_mae']:.4f}")
        print(f"   CV R²: {data['cv_r2']:.4f}")
        print(f"   Inference Time: {data['cv_inference_time_ms']:.2f}ms")
        comparison['ranking'].append({
            'rank': rank,
            'model': name,
            'cv_rmse': data['cv_rmse'],
            'cv_mae': data['cv_mae'],
            'cv_r2': data['cv_r2'],
            'inference_time_ms': data['cv_inference_time_ms']
        })
    
    # Save comparison report
    comparison_path = MODELS_DIR / 'model_comparison.json'
    with open(comparison_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\nComparison report saved to: {comparison_path}")
    
    # Recommend models for ensemble (top 2-3 with inference time < 200ms)
    print("\nRecommended Models for Ensemble:")
    recommended = [
        (name, data) for name, data in successful_models
        if data.get('cv_inference_time_ms', float('inf')) < 200
    ][:3]
    
    for name, data in recommended:
        print(f"  - {name} (RMSE: {data['cv_rmse']:.4f}, Inference: {data['cv_inference_time_ms']:.2f}ms)")
    
    return comparison


if __name__ == '__main__':
    train_all_models()

