import logo from '../../assets/logo.png';

const WelcomeSection = () => {
  return (
    <div className="header" style={{ padding: '10px', borderRadius: '20px' }}>
      <img src={logo} alt="Prenergyze" className="brand-logo" style={{ height: 'auto', maxHeight: '800px', width: 'auto', maxWidth: '100%', marginBottom: '-150px', marginTop: '-150px' }} />
    </div>
  );
};

export default WelcomeSection;
