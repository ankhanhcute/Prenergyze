const ResearchAbstract = () => {
  return (
    <div className="card">
      <div style={{ lineHeight: '1.8', color: '#d1d5db' }}>
        <p style={{ marginBottom: '20px' }}>
          <strong style={{ color: '#f59e0b' }}>Problem Statement:</strong> Accurate energy load forecasting is critical for 
          grid operators to ensure reliable electricity supply and optimize resource allocation. 
          Traditional forecasting methods often struggle with the complex relationships between 
          weather patterns and energy demand.
        </p>
        <p style={{ marginBottom: '20px' }}>
          <strong style={{ color: '#f59e0b' }}>Methodology:</strong> This system employs an advanced ensemble strategy, combining 
          predictions from Linear Regression, Random Forest, XGBoost, LightGBM, LSTM, and SARIMAX. 
          We utilize a recursive forecasting approach with dynamic lag features to capture complex cyclical patterns. 
          The ensemble uses a hybrid weighting system that prioritizes low-RMSE models while ensuring seasonal 
          models (SARIMAX) contribute to the long-term wave structure.
        </p>
        <p style={{ marginBottom: '20px' }}>
          <strong style={{ color: '#f59e0b' }}>Data Sources:</strong> The system integrates comprehensive weather metrics from 
          Open-Meteo API (temperature, humidity, precipitation, cloud cover, wind data) with 
          historical energy demand data from the U.S. Energy Information Administration (EIA).
        </p>
        <p>
          <strong style={{ color: '#f59e0b' }}>Key Features:</strong> The ensemble automatically selects optimal models based 
          on cross-validation performance metrics, providing accurate forecasts while maintaining 
          fast inference times suitable for real-time applications.
        </p>
      </div>
    </div>
  );
};

export default ResearchAbstract;

