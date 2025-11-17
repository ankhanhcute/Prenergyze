import pandas as pd
import glob
import os
from pathlib import Path

BASE_DIR = Path(os.getcwd())

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

    data_path = os.path.join(BASE_DIR, 'backend', 'data', 'processed', 'CLEAN_MERGED_DATASET.csv')
    data = pd.read_csv(data_path)

    data['date'] = pd.to_datetime(data['date'])
    data['date'] = data['date'].fillna(method='ffill')
    print(data['date'].isna().sum())


    # Sort by date to ensure chronological order
    data = data.sort_values('date').reset_index(drop=True)

    # Create time_idx - required by TimeSeriesDataSet (sequential integer index)
    data['time_idx'] = (data['date'] - data['date'].min()).dt.total_seconds() / 3600
    data['time_idx'] = data['time_idx'].astype(int)

    # Create a dummy group_id for single time series (all rows get same ID)
    data['group_id'] = 0

    # Create time-based features for static encoding (optional but helpful)
    data['year'] = data['date'].dt.year.astype(str)
    data['month'] = data['date'].dt.month.astype(str)
    data['day'] = data['date'].dt.day.astype(str)
    data['hour'] = data['date'].dt.hour.astype(str)
    data['day_of_week'] = data['date'].dt.dayofweek.astype(str)
    data['is_weekend'] = (data['date'].dt.dayofweek >= 5).astype(int).astype(str).astype('category').astype(str)

    data['grid_id'] = 'grid_1'

    data.info()

    def past_mean(series, window):
        return series.shift(1).rolling(window, min_periods=1).mean()

    def past_std(series, window):
        return series.shift(1).rolling(window, min_periods=1).std()
        
    def past_sum(series, window):
        return series.shift(1).rolling(window, min_periods=1).sum()

    # Shifting all the features to avoid data leakage
    # Since we want the current row to only contain info from the past

    # load-based
    data['load_roll_std_3h']    = past_std(data['load'], 3)
    data['load_roll_std_24h']   = past_std(data['load'], 24)
    data['load_roll_std_168h']  = past_std(data['load'], 168)
    data['load_roll_mean_24h']  = past_mean(data['load'], 24)
    data['load_roll_mean_168h'] = past_mean(data['load'], 168)

    # wind direction
    data['wind_direction_10m_roll_mean_3h'] = past_mean(data['wind_direction_10m'], 3)

    # pressure & temperature
    data['pressure_msl_roll_mean_3h']  = past_mean(data['pressure_msl'], 3)
    data['pressure_msl_roll_mean_24h'] = past_mean(data['pressure_msl'], 24)
    data['temperature_2m_roll_mean_3h']  = past_mean(data['temperature_2m'], 3)
    data['temperature_2m_roll_mean_24h'] = past_mean(data['temperature_2m'], 24)

    # apparent temp & humidity
    data['apparent_temperature_roll_mean_3h']  = past_mean(data['apparent_temperature'], 3)
    data['relative_humidity_2m_roll_mean_3h']  = past_mean(data['relative_humidity_2m'], 3)
    data['relative_humidity_2m_roll_mean_24h'] = past_mean(data['relative_humidity_2m'], 24)

    # VPD
    data['vapour_pressure_deficit_roll_mean_3h']  = past_mean(data['vapour_pressure_deficit'], 3)
    data['vapour_pressure_deficit_roll_mean_24h'] = past_mean(data['vapour_pressure_deficit'], 24)

    # clouds/wind/sun
    data['cloud_cover_roll_mean_3h']      = past_mean(data['cloud_cover'], 3)
    data['cloud_cover_low_roll_mean_3h']  = past_mean(data['cloud_cover_low'], 3)
    data['cloud_cover_mid_roll_mean_3h']  = past_mean(data['cloud_cover_mid'], 3)
    data['cloud_cover_high_roll_mean_3h'] = past_mean(data['cloud_cover_high'], 3)
    data['wind_gusts_10m_roll_mean_3h']   = past_mean(data['wind_gusts_10m'], 3)
    data['wind_speed_10m_roll_mean_3h']   = past_mean(data['wind_speed_10m'], 3)
    data['sunshine_duration_roll_mean_3h'] = past_mean(data['sunshine_duration'], 3)
    data['et0_fao_evapotranspiration_roll_mean_3h'] = past_mean(data['et0_fao_evapotranspiration'], 3)

    # precipitation (sum)
    data['precipitation_roll_sum_3h'] = past_sum(data['precipitation'], 3)

    # safe lag
    data['load_lag_1h']   = data['load'].shift(1)
    data['load_lag_2h']   = data['load'].shift(2)
    data['load_lag_3h']   = data['load'].shift(3)
    data['load_lag_24h']  = data['load'].shift(24)
    data['load_lag_25h']  = data['load'].shift(25)
    data['load_lag_168h'] = data['load'].shift(168)

    data['pressure_msl_lag_24h']            = data['pressure_msl'].shift(24)
    data['temperature_2m_lag_24h']          = data['temperature_2m'].shift(24)
    data['relative_humidity_2m_lag_1h']     = data['relative_humidity_2m'].shift(1)
    data['relative_humidity_2m_lag_24h']    = data['relative_humidity_2m'].shift(24)
    data['vapour_pressure_deficit_lag_1h']  = data['vapour_pressure_deficit'].shift(1)
    data['vapour_pressure_deficit_lag_24h'] = data['vapour_pressure_deficit'].shift(24)
    data['apparent_temperature_lag_24h']    = data['apparent_temperature'].shift(24)

    data.dropna(inplace=True)

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
    from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
    from lightning.pytorch.loggers import TensorBoardLogger
    from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
    from pytorch_forecasting.data import GroupNormalizer
    from pytorch_forecasting.metrics import QuantileLoss, MAE, RMSE
    import warnings
    warnings.filterwarnings("ignore")

    max_prediction_length = 24
    max_encoder_length = 168

    training_cutoff = data['time_idx'].max() - max_prediction_length

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

    training_dataloader = training.to_dataloader(
        train=True, 
        batch_size=batch_size, 
        num_workers=4
    )

    validation_dataloader = validation.to_dataloader(
        train=False, 
        batch_size=batch_size * 10, 
        num_workers=4
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
    logger = TensorBoardLogger('lightning_logs')

    trainer = pl.Trainer(
        max_epochs=50,
        accelerator='auto',
        devices='auto',
        gradient_clip_val=0.1,
        callbacks=[lr_logger, early_stop_callback],
        logger=logger,
        enable_progress_bar=True,
        default_root_dir= './checkpoints'
    )

    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.001,  # Start with this, can tune later
        hidden_size=64,       # Increase for larger datasets (32-128)
        attention_head_size=4, # Number of attention heads (1-4)
        dropout=0.3,          # Regularization (0.1-0.3)
        hidden_continuous_size=64,  # Should be <= hidden_size
        loss=QuantileLoss(),  # For probabilistic forecasting  
        optimizer='ranger',   # Adam optimizer variant
        reduce_on_plateau_patience=4,
        log_interval=10
    )

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

    print(f"Number of parameters in network: {tft.size() / 1e3:.1f}k")
 
    from lightning.pytorch.tuner import Tuner

    res = Tuner(trainer).lr_find(
        tft,
        train_dataloaders=training_dataloader,
        val_dataloaders=validation_dataloader,
        max_lr=0.1,
        min_lr=1e-6
    )

    print(f"Suggested learning rate: {res.suggestion()}")

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