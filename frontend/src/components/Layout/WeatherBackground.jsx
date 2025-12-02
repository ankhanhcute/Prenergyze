import { useEffect, useState } from 'react';
import { useLocation } from 'react-router-dom';
import './WeatherBackground.css';

const WeatherBackground = () => {
  const location = useLocation();
  const [weatherState, setWeatherState] = useState('night');

  useEffect(() => {
    const path = location.pathname;
    if (path === '/') setWeatherState('night');
    else if (path === '/data') setWeatherState('sunny');
    else if (path === '/forecast') setWeatherState('rainy');
    else if (path === '/models') setWeatherState('overcast');
    else setWeatherState('night'); // Default
  }, [location]);

  // Helper to generate random stars
  const renderStars = (count) => {
    const stars = [];
    for (let i = 0; i < count; i++) {
      const style = {
        left: `${Math.random() * 100}%`,
        top: `${Math.random() * 100}%`,
        animationDelay: `${Math.random() * 2}s`,
        width: `${Math.random() * 3 + 1}px`,
        height: `${Math.random() * 3 + 1}px`
      };
      stars.push(<div key={i} className="star-small" style={style}></div>);
    }
    return stars;
  };

  // Helper to generate rain drops
  const renderRain = (count) => {
    const drops = [];
    for (let i = 0; i < count; i++) {
      const style = {
        left: `${Math.random() * 100}%`,
        animationDuration: `${0.5 + Math.random() * 0.5}s`,
        animationDelay: `${Math.random() * 2}s`
      };
      drops.push(<div key={i} className="drop" style={style}></div>);
    }
    return drops;
  };

  // Helper to generate clouds
  const renderClouds = (count, isDark = false) => {
    const clouds = [];
    for (let i = 0; i < count; i++) {
      const size = 100 + Math.random() * 150;
      const style = {
        width: `${size}px`,
        height: `${size * 0.6}px`,
        top: `${Math.random() * 40}%`,
        animationDuration: `${20 + Math.random() * 40}s`,
        animationDelay: `${Math.random() * -20}s`,
        opacity: 0.7
      };
      clouds.push(
        <div 
          key={i} 
          className={`cloud ${isDark ? 'dark-cloud' : ''}`} 
          style={style}
        ></div>
      );
    }
    return clouds;
  };

  return (
    <div className="weather-bg-container">
      {/* Night Layer */}
      <div className={`weather-layer theme-night ${weatherState === 'night' ? 'active' : ''}`}>
        <div className="stars">{renderStars(50)}</div>
      </div>

      {/* Sunny Layer */}
      <div className={`weather-layer theme-sunny ${weatherState === 'sunny' ? 'active' : ''}`}>
        <div className="sun"></div>
        {renderClouds(5)}
      </div>

      {/* Rainy Layer */}
      <div className={`weather-layer theme-rainy ${weatherState === 'rainy' ? 'active' : ''}`}>
        {renderClouds(8, true)}
        {renderRain(40)}
      </div>

      {/* Overcast Layer */}
      <div className={`weather-layer theme-overcast ${weatherState === 'overcast' ? 'active' : ''}`}>
        {renderClouds(12, true)}
        <div className="fog"></div>
      </div>
    </div>
  );
};

export default WeatherBackground;

