import pandas as pd
import glob
import os
from pathlib import Path
import sys

# Add training scripts to path
sys.path.insert(0, str(Path(__file__).parent))

from data_prep import load_and_prepare_data

BASE_DIR = Path(__file__).resolve().parent.parent.parent  # This gives 'backend/'

def find_latest_checkpoint():
    # Find all .ckpt files
    all_checkpoints = glob.glob('lightning_logs/**/checkpoints/*.ckpt', recursive=True)
    
    if not all_checkpoints:
        print("No checkpoints found!")
        return None
    
    # Sort by modification time and get the latest
    latest = max(all_checkpoints, key=os.path.getmtime)
    print(f"Latest checkpoint: {latest}")
    return latest

def train(resume_checkpoint = None):
    # Use standardized data preparation
    data_path = BASE_DIR / 'data' / 'processed' / 'FEATURE_ENGINEERED_DATASET.csv'
    
    # Load data with standardized feature engineering
    # Note: We need to preserve date for TFT, so we'll reload it
    data = pd.read_csv(data_path)
    data['date'] = pd.to_datetime(data['date'])
    data = data.sort_values('date').reset_index(drop=True)
    
    # Apply standardized feature engineering (but preserve date column)
    data_prepared = load_and_prepare_data(str(data_path))
    
    # Merge back the date column from original data (matching by index after dropna)
    # Since load_and_prepare_data drops first 168 rows and NaN rows, we need to align
    original_with_date = data.copy()
    original_with_date = original_with_date.iloc[168:].reset_index(drop=True)  # Match the iloc[168:] from data_prep
    original_with_date = original_with_date.dropna().reset_index(drop=True)  # Match dropna from data_prep
    
    # Ensure same length
    if len(original_with_date) == len(data_prepared):
        data_prepared['date'] = original_with_date['date'].values
    else:
        # Fallback: recreate date from index if lengths don't match
        print("Warning: Date alignment issue, recreating date from index")
        start_date = data['date'].min()
        data_prepared['date'] = pd.date_range(start=start_date, periods=len(data_prepared), freq='H')
    
    # Create time_idx - required by TimeSeriesDataSet (sequential integer index)
    data_prepared['time_idx'] = (data_prepared['date'] - data_prepared['date'].min()).dt.total_seconds() / 3600
    data_prepared['time_idx'] = data_prepared['time_idx'].astype(int)

    # Create a dummy group_id for single time series (all rows get same ID)
    data_prepared['group_id'] = 0

    # Create time-based features for static encoding (TFT-specific, as strings)
    data_prepared['year'] = data_prepared['date'].dt.year.astype(str)
    data_prepared['month'] = data_prepared['date'].dt.month.astype(str)
    data_prepared['day'] = data_prepared['date'].dt.day.astype(str)
    data_prepared['hour'] = data_prepared['date'].dt.hour.astype(str)
    data_prepared['day_of_week'] = data_prepared['date'].dt.dayofweek.astype(str)
    data_prepared['is_weekend'] = (data_prepared['date'].dt.dayofweek >= 5).astype(int).astype(str)

    data_prepared['grid_id'] = 'grid_1'
    
    # Handle wind direction - TFT uses wind_direction_10m, but FEATURE_ENGINEERED_DATASET has sin/cos
    # If wind_direction_10m doesn't exist, we'll skip it (TFT will use wind_dir_sin/cos if available)
    if 'wind_direction_10m' not in data_prepared.columns:
        # Create wind_direction_10m from sin/cos if available
        if 'wind_dir_sin_10m' in data_prepared.columns and 'wind_dir_cos_10m' in data_prepared.columns:
            import numpy as np
            data_prepared['wind_direction_10m'] = np.arctan2(
                data_prepared['wind_dir_sin_10m'], 
                data_prepared['wind_dir_cos_10m']
            ) * 180 / np.pi
            # Add rolling mean for wind direction
            data_prepared['wind_direction_10m_roll_mean_3h'] = (
                data_prepared['wind_direction_10m'].shift(1).rolling(3, min_periods=1).mean()
            )

    data = data_prepared
    data.info()

    # Drop date column as TimeSeriesDataSet uses time_idx
    data = data.drop(['date'], axis=1)
    data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

    # Verify we have time_idx and group_id
    print(f"Data shape: {data.shape}")
    print(f"Has time_idx: {'time_idx' in data.columns}")
    print(f"Has group_id: {'group_id' in data.columns}")
    print(f"Has load: {'load' in data.columns}")
    print(f"\nFirst few columns: {list(data.columns[:10])}")

    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
    from lightning.pytorch.loggers import TensorBoardLogger
    from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
    from pytorch_forecasting.data import GroupNormalizer
    from pytorch_forecasting.metrics import QuantileLoss, MAE, RMSE
    import warnings
    warnings.filterwarnings("ignore")

    max_prediction_length = 24
    max_encoder_length = 168

    # Use proper validation split (1800 hours ~ 75 days) to match other models
    validation_hours = 1800
    training_cutoff = data['time_idx'].max() - validation_hours
    print(f"Training cutoff: {training_cutoff} (using last {validation_hours} hours for validation)")

    training = TimeSeriesDataSet(
        data[lambda x: x.time_idx <= training_cutoff],
        time_idx='time_idx',
        target='load',
        group_ids=['grid_id'], 
        min_encoder_length=max_encoder_length // 2,
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        
        static_categoricals=['grid_id'],
        static_reals=[], 
        
        time_varying_known_categoricals=['hour', 'day_of_week', 'month', 'is_weekend'],
        time_varying_known_reals=['time_idx'],
        
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=[  
            'load',
            'load_roll_std_3h',
            'load_roll_std_24h',
            'load_roll_std_168h',
            'load_roll_mean_24h',
            'load_roll_mean_168h',
            
            'wind_direction_10m',
            'wind_direction_10m_roll_mean_3h',

            'pressure_msl',
            'pressure_msl_roll_mean_3h',
            'pressure_msl_roll_mean_24h',
            'pressure_msl_lag_24h',

            'temperature_2m',
            'temperature_2m_roll_mean_3h',
            'temperature_2m_roll_mean_24h',
            'temperature_2m_lag_24h',

            'apparent_temperature',
            'apparent_temperature_roll_mean_3h',
            'apparent_temperature_lag_24h',

            'relative_humidity_2m_lag_1h',
            'relative_humidity_2m_lag_24h',

            'relative_humidity_2m',
            'relative_humidity_2m_roll_mean_3h',
            'relative_humidity_2m_roll_mean_24h',

            'vapour_pressure_deficit',
            'vapour_pressure_deficit_roll_mean_3h',
            'vapour_pressure_deficit_roll_mean_24h',
            'vapour_pressure_deficit_lag_1h',
            'vapour_pressure_deficit_lag_24h',

            'cloud_cover',
            'cloud_cover_roll_mean_3h',

            'cloud_cover_low',
            'cloud_cover_low_roll_mean_3h',

            'cloud_cover_mid',
            'cloud_cover_mid_roll_mean_3h',

            'cloud_cover_high',
            'cloud_cover_high_roll_mean_3h',

            'wind_gusts_10m',
            'wind_gusts_10m_roll_mean_3h',

            'wind_speed_10m',
            'wind_speed_10m_roll_mean_3h',

            'sunshine_duration',
            'sunshine_duration_roll_mean_3h',

            'et0_fao_evapotranspiration',
            'et0_fao_evapotranspiration_roll_mean_3h',

            'precipitation',
            'precipitation_roll_sum_3h',

            'load_lag_1h',
            'load_lag_2h',
            'load_lag_3h',
            'load_lag_24h',
            'load_lag_25h',
            'load_lag_168h',

        ],
        
        target_normalizer=GroupNormalizer(
            groups=['grid_id'], 
            transformation='softplus'
        ),
        
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True
    )

    validation = TimeSeriesDataSet.from_dataset(
        training, 
        data, 
        predict=True, 
        stop_randomization=True
    )

    batch_size = 64 

    # Use num_workers=0 on Windows to avoid multiprocessing issues (spawn vs fork)
    import platform
    num_workers = 0 if platform.system() == 'Windows' else 4
    
    training_dataloader = training.to_dataloader(
        train=True, 
        batch_size=batch_size, 
        num_workers=num_workers
    )

    validation_dataloader = validation.to_dataloader(
        train=False, 
        batch_size=batch_size * 10, 
        num_workers=num_workers
    )

    pl.seed_everything(42)

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=1e-4,
        patience=10,
        verbose=False,
        mode='min'
    )

    lr_logger = LearningRateMonitor()
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        filename='best-{epoch:02d}-{val_loss:.2f}'
    )
    logger = TensorBoardLogger('lightning_logs')

    trainer = pl.Trainer(
        max_epochs=50,
        accelerator='auto',
        devices='auto',
        gradient_clip_val=0.1,
        callbacks=[lr_logger, early_stop_callback, checkpoint_callback],
        logger=logger,
        enable_progress_bar=True,
        default_root_dir= './checkpoints'
    )

    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.001,  # Initial LR, will be updated by LR finder
        hidden_size=64,       # Increase for larger datasets (32-128)
        attention_head_size=4, # Number of attention heads (1-4)
        dropout=0.3,          # Regularization (0.1-0.3)
        hidden_continuous_size=64,  # Should be <= hidden_size
        loss=QuantileLoss(),  # For probabilistic forecasting  
        optimizer='ranger',   # Adam optimizer variant
        reduce_on_plateau_patience=4,
        log_interval=10
    )

    # Run LR finder BEFORE training (not after) to find optimal learning rate
    # Skip LR finder if resuming from checkpoint (model already has trained LR)
    if not resume_checkpoint:
        print("Running learning rate finder...")
        from lightning.pytorch.tuner import Tuner
        
        res = Tuner(trainer).lr_find(
            tft,
            train_dataloaders=training_dataloader,
            val_dataloaders=validation_dataloader,
            max_lr=0.1,
            min_lr=1e-6
        )
        
        suggested_lr = res.suggestion()
        if suggested_lr is not None and suggested_lr > 0:
            tft.learning_rate = suggested_lr
            print(f"Using suggested learning rate: {suggested_lr:.6f}")
        else:
            print(f"LR finder didn't suggest a rate, keeping default: {tft.learning_rate}")
    else:
        print("Skipping LR finder (resuming from checkpoint)")

    print(f"Number of parameters in network: {tft.size() / 1e3:.1f}k")

    # Train the model with optimal learning rate
    if resume_checkpoint:
        print(f"Resuming training from: {resume_checkpoint}")
        trainer.fit(
            tft,
            train_dataloaders=training_dataloader,
            val_dataloaders=validation_dataloader,
            ckpt_path=resume_checkpoint
        )
    else:
        trainer.fit(
            tft,
            train_dataloaders=training_dataloader,
            val_dataloaders=validation_dataloader
        )

    best_model_path = trainer.checkpoint_callback.best_model_path
    best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

    predictions = best_tft.predict(
        validation_dataloader, 
        return_y=True, 
        trainer_kwargs=dict(accelerator='auto')
    )

    mae = MAE()(predictions.output, predictions.y)
    print(f"Validation MAE: {mae:.2f}")

    rmse = RMSE()(predictions.output, predictions.y)
    print(f"Validation RMSE: {rmse:.2f}")

    raw_predictions = best_tft.predict(
        validation_dataloader,
        mode='raw',
        return_x=True,
        trainer_kwargs=dict(accelerator='auto')
    )

    for idx in range(5):
        best_tft.plot_prediction(
            raw_predictions.x,
            raw_predictions.output,
            idx=idx,
            add_loss_to_title=True
        )

if __name__ == '__main__':
    if input("Resume from last checkpoint? ") == "Y":
        resume_from = find_latest_checkpoint()
        if resume_from:
            print(f"Resuming from: {resume_from}")
            results = train(resume_checkpoint=resume_from)
        else:
            print("No checkpoint found, starting fresh training.")
            results = train()
    else:
        print("Starting fresh training.")
        results = train()
    print(results)