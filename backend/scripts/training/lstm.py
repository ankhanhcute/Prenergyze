"""
LSTM model training script for time series forecasting.
"""
import os
import pickle
import time
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
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


class TimeSeriesDataset(Dataset):
    """Dataset for LSTM time series forecasting."""
    
    def __init__(self, X, y, sequence_length=24):
        self.X = X
        self.y = y
        self.sequence_length = sequence_length
    
    def __len__(self):
        return len(self.X) - self.sequence_length + 1
    
    def __getitem__(self, idx):
        X_seq = self.X[idx:idx + self.sequence_length]
        y_val = self.y[idx + self.sequence_length - 1]
        return torch.FloatTensor(X_seq), torch.FloatTensor([y_val])


class LSTMModel(nn.Module):
    """LSTM model for time series forecasting."""
    
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # Take the last output
        last_output = lstm_out[:, -1, :]
        last_output = self.dropout(last_output)
        output = self.fc(last_output)
        return output


def train_lstm_epoch(model, dataloader, criterion, optimizer, device):
    """Train one epoch of LSTM."""
    model.train()
    total_loss = 0
    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate_lstm(model, dataloader, criterion, device):
    """Evaluate LSTM model."""
    model.eval()
    total_loss = 0
    predictions = []
    targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            total_loss += loss.item()
            
            predictions.extend(y_pred.cpu().numpy().flatten())
            targets.extend(y_batch.cpu().numpy().flatten())
    
    return total_loss / len(dataloader), np.array(predictions), np.array(targets)


def train_lstm(
    evaluate_cv: bool = True,
    save_model: bool = True,
    sequence_length: int = 24,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
    learning_rate: float = 0.001,
    batch_size: int = 32,
    epochs: int = 50,
    patience: int = 10
):
    """
    Train LSTM model with cross-validation evaluation.
    
    Args:
        evaluate_cv: Whether to perform cross-validation evaluation
        save_model: Whether to save the trained model
        sequence_length: Length of input sequences
        hidden_size: Number of LSTM hidden units
        num_layers: Number of LSTM layers
        dropout: Dropout rate
        learning_rate: Learning rate
        batch_size: Batch size
        epochs: Maximum number of epochs
        patience: Early stopping patience
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
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
            X_train, X_test = X.iloc[train_index].values, X.iloc[test_index].values
            y_train, y_test = y.iloc[train_index].values, y.iloc[test_index].values
            
            # Scale features
            scaler_X = StandardScaler()
            X_train_scaled = scaler_X.fit_transform(X_train)
            X_test_scaled = scaler_X.transform(X_test)
            
            # Scale target
            scaler_y = StandardScaler()
            y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
            y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
            
            # Create datasets
            train_dataset = TimeSeriesDataset(X_train_scaled, y_train_scaled, sequence_length)
            test_dataset = TimeSeriesDataset(X_test_scaled, y_test_scaled, sequence_length)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            
            # Initialize model
            model = LSTMModel(
                input_size=X_train_scaled.shape[1],
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout
            ).to(device)
            
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            
            # Training with early stopping
            best_val_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(epochs):
                train_loss = train_lstm_epoch(model, train_loader, criterion, optimizer, device)
                val_loss, _, _ = evaluate_lstm(model, test_loader, criterion, device)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = model.state_dict().copy()
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch + 1}")
                    break
            
            # Load best model
            model.load_state_dict(best_model_state)
            
            # Evaluate on test set
            test_loss, y_pred_scaled, y_test_scaled_eval = evaluate_lstm(model, test_loader, criterion, device)
            
            # Inverse transform predictions
            y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            y_test_actual = scaler_y.inverse_transform(y_test_scaled_eval.reshape(-1, 1)).flatten()
            
            rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
            mae = mean_absolute_error(y_test_actual, y_pred)
            r2 = r2_score(y_test_actual, y_pred)
            
            # Measure inference time
            model.eval()
            start_time = time.time()
            with torch.no_grad():
                sample_batch = next(iter(test_loader))
                X_sample = sample_batch[0][:100].to(device)
                _ = model(X_sample)
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
    
    X_train_full = X_train_full.values
    X_test_full = X_test_full.values
    y_train_full = y_train_full.values
    y_test_full = y_test_full.values
    
    # Scale features
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train_full)
    X_test_scaled = scaler_X.transform(X_test_full)
    
    # Scale target
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train_full.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test_full.reshape(-1, 1)).flatten()
    
    # Create datasets
    train_dataset = TimeSeriesDataset(X_train_scaled, y_train_scaled, sequence_length)
    test_dataset = TimeSeriesDataset(X_test_scaled, y_test_scaled, sequence_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = LSTMModel(
        input_size=X_train_scaled.shape[1],
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training with early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        train_loss = train_lstm_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, _, _ = evaluate_lstm(model, test_loader, criterion, device)
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1}/{epochs}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Evaluate on test set
    test_loss, y_pred_scaled, y_test_scaled_eval = evaluate_lstm(model, test_loader, criterion, device)
    
    # Inverse transform predictions
    y_pred_test = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_test_actual = scaler_y.inverse_transform(y_test_scaled_eval.reshape(-1, 1)).flatten()
    
    test_rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_test))
    test_mae = mean_absolute_error(y_test_actual, y_pred_test)
    test_r2 = r2_score(y_test_actual, y_pred_test)
    
    results['test_rmse'] = test_rmse
    results['test_mae'] = test_mae
    results['test_r2'] = test_r2
    results['feature_names'] = list(X.columns)
    results['sequence_length'] = sequence_length
    results['hidden_size'] = hidden_size
    results['num_layers'] = num_layers
    results['dropout'] = dropout
    
    print(f"\nTest Set Results:")
    print(f"  RMSE: {test_rmse:.4f}")
    print(f"  MAE: {test_mae:.4f}")
    print(f"  R²: {test_r2:.4f}")
    
    if save_model:
        print("\nSaving model...")
        model_path = MODELS_DIR / 'lstm.pth'
        scaler_X_path = MODELS_DIR / 'lstm_scaler_X.pkl'
        scaler_y_path = MODELS_DIR / 'lstm_scaler_y.pkl'
        metadata_path = MODELS_DIR / 'lstm_metadata.pkl'
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': {
                'input_size': X_train_scaled.shape[1],
                'hidden_size': hidden_size,
                'num_layers': num_layers,
                'dropout': dropout
            }
        }, model_path)
        
        with open(scaler_X_path, 'wb') as f:
            pickle.dump(scaler_X, f)
        
        with open(scaler_y_path, 'wb') as f:
            pickle.dump(scaler_y, f)
        
        with open(metadata_path, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"  Model saved to: {model_path}")
        print(f"  Scaler X saved to: {scaler_X_path}")
        print(f"  Scaler y saved to: {scaler_y_path}")
        print(f"  Metadata saved to: {metadata_path}")
    
    return model, scaler_X, scaler_y, results


if __name__ == '__main__':
    train_lstm(
        evaluate_cv=True,
        save_model=True,
        sequence_length=24,
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
        learning_rate=0.001,
        batch_size=32,
        epochs=50,
        patience=10
    )

