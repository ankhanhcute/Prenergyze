import HistoricalDataChart from '../components/VisualizationPage/HistoricalDataChart';
import DataVisualizations from '../components/VisualizationPage/DataVisualizations';

const DataPage = () => {
  return (
    <div className="main-content" style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      <div style={{ height: '500px' }}>
        <HistoricalDataChart />
      </div>
      <div style={{ height: '600px' }}>
        <DataVisualizations />
      </div>
    </div>
  );
};

export default DataPage;

