import { useState, useCallback } from 'react';
import WeatherForecastSelector from '../components/VisualizationPage/WeatherForecastSelector';
import ForecastChart from '../components/VisualizationPage/ForecastChart';
import DailyForecastChart from '../components/VisualizationPage/DailyForecastChart';

const ForecastPage = () => {
  const [weatherData, setWeatherData] = useState(null);
  const [forecastData, setForecastData] = useState(null);

  const handleWeatherDataFetched = useCallback((data) => {
    setWeatherData(data);
  }, []);

  const handleForecastGenerated = useCallback((data) => {
    setForecastData(data);
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
        <ForecastChart 
          weatherData={weatherData} 
          onForecastGenerated={handleForecastGenerated}
        />
      </div>
      <div>
        <DailyForecastChart forecastData={forecastData} />
      </div>
    </div>
  );
};

export default ForecastPage;
