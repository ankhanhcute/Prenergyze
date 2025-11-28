import { Link } from 'react-router-dom';

const NavigationCard = () => {
  return (
    <div className="card" style={{ textAlign: 'center', padding: '40px' }}>
      <h2 style={{ marginBottom: '20px' }}>Explore Data & Predictions</h2>
      <p style={{ marginBottom: '30px', color: '#666', fontSize: '16px' }}>
        View interactive visualizations of historical data, explore correlations between 
        weather variables and energy load, and generate forecasted predictions using our 
        ML models with real-time weather forecasts.
      </p>
      <Link to="/visualizations" className="btn" style={{ textDecoration: 'none', display: 'inline-block' }}>
        Go to Visualizations →
      </Link>
    </div>
  );
};

export default NavigationCard;

