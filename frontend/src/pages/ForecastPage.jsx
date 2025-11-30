import { useState, useCallback } from 'react';
import WeatherForecastSelector from '../components/VisualizationPage/WeatherForecastSelector';
import ForecastChart from '../components/VisualizationPage/ForecastChart';

const ForecastPage = () => {
  const [weatherData, setWeatherData] = useState(null);

  const handleWeatherDataFetched = useCallback((data) => {
    setWeatherData(data);
  }, []);

  return (
    <div className="main-content" style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      <div>
        <WeatherForecastSelector 
          onWeatherDataFetched={handleWeatherDataFetched}
          autoFetch={true}
        />
      </div>
      <div>
        <ForecastChart weatherData={weatherData} />
      </div>
    </div>
  );
};

export default ForecastPage;

