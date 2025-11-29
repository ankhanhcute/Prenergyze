import { Link, useLocation } from 'react-router-dom';
import logo from '../../assets/logo.png';
import './Navbar.css';

const Navbar = () => {
  const location = useLocation();

  return (
    <nav className="navbar">
      <div className="navbar-container">
        <Link to="/" className="navbar-brand">
          <img src={logo} alt="Prenergyze" className="navbar-logo" />
        </Link>
        <div className="navbar-links">
          <Link 
            to="/" 
            className={location.pathname === '/' ? 'active' : ''}
          >
            Home
          </Link>
          <Link 
            to="/visualizations" 
            className={location.pathname === '/visualizations' ? 'active' : ''}
          >
            Visualizations
          </Link>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;

