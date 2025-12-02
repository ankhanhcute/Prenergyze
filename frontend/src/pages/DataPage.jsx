import HistoricalDataChart from '../components/VisualizationPage/HistoricalDataChart';
import DataVisualizations from '../components/VisualizationPage/DataVisualizations';

const DataPage = () => {
  return (
    <div className="main-content" style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      <div>
        <HistoricalDataChart />
      </div>
      <div>
        <DataVisualizations />
      </div>
    </div>
  );
};

export default DataPage;

