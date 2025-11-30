import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Layout from './components/Layout/Layout';
import EntryPage from './pages/EntryPage';
import DataPage from './pages/DataPage';
import ForecastPage from './pages/ForecastPage';
import ModelInfoPage from './pages/ModelInfoPage';

function App() {
  return (
    <Router>
      <Layout>
        <Routes>
          <Route path="/" element={<EntryPage />} />
          <Route path="/data" element={<DataPage />} />
          <Route path="/forecast" element={<ForecastPage />} />
          <Route path="/models" element={<ModelInfoPage />} />
        </Routes>
      </Layout>
    </Router>
  );
}

export default App;

