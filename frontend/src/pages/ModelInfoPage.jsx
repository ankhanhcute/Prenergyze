import { useState, useEffect } from 'react';
import { getModelInfo } from '../services/api';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  Cell
} from 'recharts';

const ModelInfoPage = () => {
  const [modelInfo, setModelInfo] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const [selectedModels, setSelectedModels] = useState(() => {
    try {
      const saved = localStorage.getItem('prenergyze_selected_models');
      return saved ? JSON.parse(saved) : [];
    } catch (e) {
      return [];
    }
  });

  useEffect(() => {
    const fetchModelInfo = async () => {
      try {
        setLoading(true);
        const data = await getModelInfo();
        
        // Robust check for data structure
        if (!data || !data.model_metadata) {
          // If data is missing or malformed, throw to trigger error state
          // However, if data is just empty, we handle it gracefully in render
          console.warn("Model info data structure might be unexpected:", data);
        }
        
        setModelInfo(data);
        setError(null);
      } catch (err) {
        console.error("Error fetching model info:", err);
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchModelInfo();
  }, []);

  // Save selected models to localStorage whenever they change
  useEffect(() => {
    localStorage.setItem('prenergyze_selected_models', JSON.stringify(selectedModels));
  }, [selectedModels]);

  const toggleModelSelection = (modelName) => {
    const originalName = modelName.toLowerCase().replace(/ /g, '_');
    
    setSelectedModels(prev => {
      if (prev.includes(originalName)) {
        return prev.filter(m => m !== originalName);
      } else {
        if (prev.length >= 3) {
          // Max 3 models logic can be handled here if we want to strictly prevent it,
          // or we can just allow toggle and disable others in UI.
          // Let's prevent adding if 3 are already selected
          return prev;
        }
        return [...prev, originalName];
      }
    });
  };

  if (loading) {
    return (
      <div className="container">
        <div className="loading">
          <div className="spinner"></div>
          Loading model information...
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="container">
        <div className="error">Error: {error}</div>
      </div>
    );
  }

  if (!modelInfo || !modelInfo.model_metadata) {
    return (
      <div className="container">
        <div className="error">No model information available.</div>
      </div>
    );
  }

  const { model_metadata } = modelInfo;
  
  // Prepare data for comparison chart
  // Safely handle potentially missing or infinity values
  const comparisonData = Object.entries(model_metadata)
    .filter(([_, metrics]) => metrics && (metrics.status === 'success' || metrics.cv_rmse !== undefined))
    .map(([name, metrics]) => ({
      name: name.replace(/_/g, ' ').toUpperCase(),
      rmse: typeof metrics.test_rmse === 'number' ? metrics.test_rmse : (metrics.cv_rmse || 0),
      mae: (typeof metrics.test_mae === 'number' && isFinite(metrics.test_mae)) ? metrics.test_mae : (metrics.cv_mae || null),
      r2: (typeof metrics.test_r2 === 'number' && isFinite(metrics.test_r2)) ? metrics.test_r2 : (metrics.cv_r2 || null),
      inference_time: (metrics.inference_time_ms || metrics.cv_inference_time_ms || 0)
    }))
    .sort((a, b) => a.rmse - b.rmse);

  // Colors for models
  const COLORS = ['#8b5cf6', '#f59e0b', '#3b82f6', '#10b981', '#ef4444', '#ec4899'];

  return (
    <div className="container">
      <div className="header">
        <h1>Model Information & Selection</h1>
        <p>View performance metrics and select up to 3 models for your custom ensemble forecast.</p>
        {selectedModels.length > 0 && (
            <div style={{ marginTop: '10px', padding: '10px', background: 'rgba(16, 185, 129, 0.1)', borderRadius: '8px', display: 'inline-block' }}>
              <span style={{ color: '#34d399', fontWeight: 'bold' }}>
                {selectedModels.length}/3 Models Selected
              </span>
            </div>
        )}
      </div>

      <div className="main-content" style={{ gridTemplateColumns: '1fr' }}>
        {/* Performance Comparison Chart */}
        <div className="card">
          <h2>Model Performance Comparison (RMSE)</h2>
          <p style={{ color: '#9ca3af', marginBottom: '20px' }}>
            Root Mean Square Error (RMSE) on test set. Lower is better.
          </p>
          <div style={{ height: '400px', width: '100%' }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={comparisonData}
                layout="vertical"
                margin={{ top: 5, right: 30, left: 40, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" horizontal={false} />
                <XAxis type="number" stroke="#9ca3af" />
                <YAxis 
                  type="category" 
                  dataKey="name" 
                  stroke="#9ca3af" 
                  width={120}
                  tick={{ fontSize: 12 }}
                />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#1f2937', borderColor: '#374151' }}
                  cursor={{ fill: 'rgba(255,255,255,0.05)' }}
                />
                <Legend />
                <Bar dataKey="rmse" name="RMSE (MW)" fill="#8b5cf6" radius={[0, 4, 4, 0]}>
                  {comparisonData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Model Details Grid */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: '24px' }}>
          {comparisonData.map((model, index) => {
            const originalName = model.name.toLowerCase().replace(/ /g, '_');
            const isSelected = selectedModels.includes(originalName);
            const isDisabled = !isSelected && selectedModels.length >= 3;

            return (
            <div key={model.name} className="card" style={{ 
              borderTop: `4px solid ${COLORS[index % COLORS.length]}`,
              position: 'relative',
              opacity: isDisabled ? 0.7 : 1,
              transition: 'all 0.3s ease'
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start' }}>
                <h3 style={{ marginTop: 0, color: '#e5e7eb' }}>{model.name}</h3>
                <label style={{ cursor: isDisabled ? 'not-allowed' : 'pointer' }}>
                  <input 
                    type="checkbox" 
                    checked={isSelected}
                    onChange={() => toggleModelSelection(model.name)}
                    disabled={isDisabled}
                    style={{ transform: 'scale(1.5)', cursor: isDisabled ? 'not-allowed' : 'pointer' }}
                  />
                </label>
              </div>
              
              <div style={{ marginTop: '15px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '10px', borderBottom: '1px solid rgba(255,255,255,0.1)', paddingBottom: '5px' }}>
                  <span style={{ color: '#9ca3af' }}>Test RMSE:</span>
                  <span style={{ fontWeight: 'bold', color: '#e5e7eb' }}>{model.rmse.toFixed(2)} MW</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '10px', borderBottom: '1px solid rgba(255,255,255,0.1)', paddingBottom: '5px' }}>
                  <span style={{ color: '#9ca3af' }}>Test MAE:</span>
                  <span style={{ fontWeight: 'bold', color: '#e5e7eb' }}>{model.mae !== null ? model.mae.toFixed(2) : 'N/A'} MW</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '10px', borderBottom: '1px solid rgba(255,255,255,0.1)', paddingBottom: '5px' }}>
                  <span style={{ color: '#9ca3af' }}>R² Score:</span>
                  <span style={{ fontWeight: 'bold', color: model.r2 > 0 ? '#10b981' : '#ef4444' }}>
                    {model.r2 !== null ? model.r2.toFixed(4) : 'N/A'}
                  </span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '10px' }}>
                  <span style={{ color: '#9ca3af' }}>Inference Time:</span>
                  <span style={{ fontWeight: 'bold', color: '#e5e7eb' }}>{model.inference_time.toFixed(4)} ms</span>
                </div>
              </div>

              <div style={{ marginTop: '15px', padding: '10px', background: 'rgba(255,255,255,0.05)', borderRadius: '8px', fontSize: '13px' }}>
                <p style={{ margin: 0, color: '#d1d5db' }}>
                  <strong>Description:</strong> {getModelDescription(model.name)}
                </p>
              </div>
            </div>
          );
          })}
        </div>
      </div>
    </div>
  );
};

const getModelDescription = (name) => {
  const normalized = name.toLowerCase();
  if (normalized.includes('lstm')) {
    return "Long Short-Term Memory network. Deep learning model effective at capturing long-term temporal dependencies in time-series data.";
  }
  if (normalized.includes('sarimax')) {
    return "Seasonal AutoRegressive Integrated Moving Average with eXogenous regressors. Statistical model explicitly designed to capture seasonality and external weather factors.";
  }
  if (normalized.includes('random forest')) {
    return "Ensemble learning method using multiple decision trees. Robust to outliers and captures non-linear relationships well.";
  }
  if (normalized.includes('xgboost')) {
    return "eXtreme Gradient Boosting. Highly efficient gradient boosting algorithm known for high performance and speed in structured data.";
  }
  if (normalized.includes('lightgbm')) {
    return "Light Gradient Boosting Machine. Faster and more efficient than standard GBMs, capable of handling large datasets with lower memory usage.";
  }
  if (normalized.includes('linear regression')) {
    return "Simple baseline model assuming linear relationship between features and load. Fast but limited in capturing complex patterns.";
  }
  return "Machine learning model for load forecasting.";
};

export default ModelInfoPage;
