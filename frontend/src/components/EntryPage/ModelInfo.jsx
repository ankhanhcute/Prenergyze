import { useState, useEffect } from 'react';
import { getModelInfo } from '../../services/api';

const ModelInfo = () => {
  const [modelInfo, setModelInfo] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchModelInfo = async () => {
      try {
        setLoading(true);
        const data = await getModelInfo();
        setModelInfo(data);
        setError(null);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchModelInfo();
  }, []);

  if (loading) {
    return (
      <div className="card">
        <h2>Model Information</h2>
        <div className="loading">
          <div className="spinner"></div>
          Loading model information...
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="card">
        <h2>Model Information</h2>
        <div className="error">Failed to load model information: {error}</div>
      </div>
    );
  }

  if (!modelInfo) {
    return null;
  }

  const { available_models, ensemble_models, model_metadata } = modelInfo;

  return (
    <div className="card">
      <h2>Model Information</h2>
      
      <div style={{ marginBottom: '20px' }}>
        <h3 style={{ fontSize: '16px', marginBottom: '10px', color: '#333' }}>
          Available Models ({available_models?.length || 0})
        </h3>
        <ul className="model-list">
          {available_models?.map((model) => {
            const metadata = model_metadata?.[model];
            const rmse = metadata?.cv_rmse;
            const r2 = metadata?.cv_r2;
            return (
              <li key={model}>
                <strong>{model}</strong>
                {rmse && ` - RMSE: ${rmse.toFixed(2)}`}
                {r2 && ` | R²: ${r2.toFixed(3)}`}
              </li>
            );
          })}
        </ul>
      </div>

      {ensemble_models && ensemble_models.length > 0 && (
        <div style={{ marginBottom: '20px' }}>
          <h3 style={{ fontSize: '16px', marginBottom: '10px', color: '#333' }}>
            Ensemble Models (Optimal Selection)
          </h3>
          <ul className="model-list">
            {ensemble_models.map((model) => {
              const metadata = model_metadata?.[model];
              const rmse = metadata?.cv_rmse;
              return (
                <li key={model}>
                  <strong>{model}</strong>
                  {rmse && ` - RMSE: ${rmse.toFixed(2)}`}
                </li>
              );
            })}
          </ul>
        </div>
      )}

      <div style={{ marginTop: '20px', padding: '15px', background: '#f3f4f6', borderRadius: '5px' }}>
        <p style={{ fontSize: '14px', color: '#555', margin: 0 }}>
          <strong>Ensemble Strategy:</strong> The system uses inverse RMSE weighting to combine 
          predictions from the top-performing models, automatically selecting the optimal subset 
          based on cross-validation performance and inference time constraints.
        </p>
      </div>
    </div>
  );
};

export default ModelInfo;

