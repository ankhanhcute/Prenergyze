import Navbar from './Navbar';
import '../../styles/components.css';

const Layout = ({ children }) => {
  return (
    <div className="container">
      <Navbar />
      {children}
    </div>
  );
};

export default Layout;

