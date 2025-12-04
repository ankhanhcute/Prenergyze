import { useState, useCallback, useMemo } from 'react';
import WeatherForecastSelector from '../components/VisualizationPage/WeatherForecastSelector';
import ForecastChart from '../components/VisualizationPage/ForecastChart';
import DailyForecastChart from '../components/VisualizationPage/DailyForecastChart';
import PricingSummary from '../components/VisualizationPage/PricingSummary';
import { calculatePricingSummary } from '../utils/pricingModel';

const ForecastPage = () => {
  const [weatherData, setWeatherData] = useState(null);
  const [forecastData, setForecastData] = useState(null);

  const handleWeatherDataFetched = useCallback((data) => {
    setWeatherData(data);
  }, []);

  const handleForecastGenerated = useCallback((data) => {
    setForecastData(data);
  }, []);

  const pricingMetrics = useMemo(() => {
    if (!forecastData || !forecastData.forecast || !weatherData || weatherData.length === 0) {
      return null;
    }

    // Create forecast points array for pricing calculation
    // Assuming forecast array aligns with weather data array starting from now
    // We need to align them. ForecastChart does this, but we need raw data here.
    // Simplified alignment: take the first N weather points where N is forecast length
    
    const points = forecastData.forecast.map((load, index) => {
      // Fallback if weather data doesn't align perfectly length-wise
      const weatherPoint = weatherData[index] || weatherData[weatherData.length - 1];
      return {
        load,
        date: weatherPoint?.date || new Date().toISOString()
      };
    });

    return calculatePricingSummary(points);
  }, [forecastData, weatherData]);

  return (
    <div className="main-content" style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      <div>
        <WeatherForecastSelector 
          onWeatherDataFetched={handleWeatherDataFetched}
          autoFetch={true}
        />
      </div>
      
      {/* Pricing Summary - Shown when forecast is available */}
      {pricingMetrics && (
        <div>
          <PricingSummary metrics={pricingMetrics} />
        </div>
      )}

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
