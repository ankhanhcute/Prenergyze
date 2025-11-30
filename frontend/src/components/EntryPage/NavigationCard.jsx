import { Link } from 'react-router-dom';

const NavigationCard = () => {
  return (
    <div className="card" style={{ textAlign: 'center', padding: '40px' }}>
      <h2 style={{ marginBottom: '20px' }}>Explore Data & Predictions</h2>
      <p style={{ marginBottom: '30px', color: '#9ca3af', fontSize: '16px', lineHeight: '1.6' }}>
        View interactive visualizations of historical data, explore correlations between 
        weather variables and energy load, and generate forecasted predictions using our 
        ML models with real-time weather forecasts.
      </p>
      <Link to="/visualizations" className="btn" style={{ textDecoration: 'none', display: 'inline-block' }}>
        Go to Data →
      </Link>
    </div>
  );
};

export default NavigationCard;

