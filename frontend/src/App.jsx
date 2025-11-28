import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Layout from './components/Layout/Layout';
import EntryPage from './pages/EntryPage';
import VisualizationPage from './pages/VisualizationPage';

function App() {
  return (
    <Router>
      <Layout>
        <Routes>
          <Route path="/" element={<EntryPage />} />
          <Route path="/visualizations" element={<VisualizationPage />} />
        </Routes>
      </Layout>
    </Router>
  );
}

export default App;

