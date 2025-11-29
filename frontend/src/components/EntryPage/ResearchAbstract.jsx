const ResearchAbstract = () => {
  return (
    <div className="card">
      <h2>Research Abstract</h2>
      <div style={{ lineHeight: '1.8', color: '#d1d5db' }}>
        <p style={{ marginBottom: '20px' }}>
          <strong style={{ color: '#a78bfa' }}>Problem Statement:</strong> Accurate energy load forecasting is critical for 
          grid operators to ensure reliable electricity supply and optimize resource allocation. 
          Traditional forecasting methods often struggle with the complex relationships between 
          weather patterns and energy demand.
        </p>
        <p style={{ marginBottom: '20px' }}>
          <strong style={{ color: '#a78bfa' }}>Methodology:</strong> This system employs an ensemble approach combining 
          predictions from five machine learning models: Linear Regression, Random Forest, 
          XGBoost, LightGBM, and LSTM. The ensemble uses inverse RMSE weighting to combine 
          predictions from the top-performing models, balancing accuracy and inference speed.
        </p>
        <p style={{ marginBottom: '20px' }}>
          <strong style={{ color: '#a78bfa' }}>Data Sources:</strong> The system integrates comprehensive weather metrics from 
          Open-Meteo API (temperature, humidity, precipitation, cloud cover, wind data) with 
          historical energy demand data from the U.S. Energy Information Administration (EIA).
        </p>
        <p>
          <strong style={{ color: '#a78bfa' }}>Key Features:</strong> The ensemble automatically selects optimal models based 
          on cross-validation performance metrics, providing accurate forecasts while maintaining 
          fast inference times suitable for real-time applications.
        </p>
      </div>
    </div>
  );
};

export default ResearchAbstract;

