import Navbar from './Navbar';
import WeatherBackground from './WeatherBackground';
import '../../styles/components.css';

const Layout = ({ children }) => {
  return (
    <div className="container">
      <WeatherBackground />
      <Navbar />
      {children}
    </div>
  );
};

export default Layout;

