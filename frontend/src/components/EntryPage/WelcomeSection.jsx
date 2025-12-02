import logo from '../../assets/logo.png';

const WelcomeSection = () => {
  return (
    <div className="header">
      <img src={logo} alt="Prenergyze" className="brand-logo" style={{ height: 'auto', maxHeight: '1000px', width: 'auto', maxWidth: '100%', marginBottom: '-150px', marginTop: '-200px' }} />
      <p style={{ marginTop: '0px', fontSize: '24px', fontWeight: '500', color: '#e5e7eb' }}>
        Energy Load Forecasting System Powered by Machine Learning
      </p>
      <p style={{ marginTop: '0px', fontSize: '16px', color: '#9ca3af', lineHeight: '1.6' }}>
        Predict electricity grid demand based on weather data using an ensemble of multiple ML models
      </p>
    </div>
  );
};

export default WelcomeSection;

