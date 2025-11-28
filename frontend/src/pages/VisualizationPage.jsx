import { useState } from 'react';
import HistoricalDataChart from '../components/VisualizationPage/HistoricalDataChart';
import CorrelationHeatmap from '../components/VisualizationPage/CorrelationHeatmap';
import WeatherForecastSelector from '../components/VisualizationPage/WeatherForecastSelector';
import ForecastChart from '../components/VisualizationPage/ForecastChart';

const VisualizationPage = () => {
  const [weatherData, setWeatherData] = useState(null);

  const handleWeatherDataFetched = (data) => {
    setWeatherData(data);
  };

  return (
    <>
      <HistoricalDataChart />
      <CorrelationHeatmap />
      <WeatherForecastSelector onWeatherDataFetched={handleWeatherDataFetched} />
      <ForecastChart weatherData={weatherData} />
    </>
  );
};

export default VisualizationPage;

