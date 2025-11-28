/**
 * Open-Meteo API client for fetching weather forecast data
 */

const METEO_BASE_URL = 'https://api.open-meteo.com/v1/forecast';

/**
 * Fetch weather forecast from Open-Meteo API
 * @param {number} lat - Latitude
 * @param {number} lon - Longitude
 * @param {string} startDate - Start date in YYYY-MM-DD format
 * @param {string} endDate - End date in YYYY-MM-DD format
 * @returns {Promise<Array>} Array of weather data objects
 */
export const fetchWeatherForecast = async (lat, lon, startDate, endDate) => {
  try {
    const hourlyParams = [
      'temperature_2m',
      'apparent_temperature',
      'relative_humidity_2m',
      'vapour_pressure_deficit',
      'pressure_msl',
      'precipitation',
      'cloud_cover',
      'cloud_cover_low',
      'cloud_cover_mid',
      'cloud_cover_high',
      'et0_fao_evapotranspiration',
      'sunshine_duration',
      'wind_speed_10m',
      'wind_gusts_10m',
      'wind_direction_10m',
    ];

    const params = new URLSearchParams({
      latitude: lat.toString(),
      longitude: lon.toString(),
      start_date: startDate,
      end_date: endDate,
      hourly: hourlyParams.join(','),
    });

    const response = await fetch(`${METEO_BASE_URL}?${params.toString()}`);
    
    if (!response.ok) {
      throw new Error(`Weather API error: ${response.status} ${response.statusText}`);
    }

    const data = await response.json();
    
    if (!data.hourly) {
      throw new Error('Invalid response from weather API');
    }

    // Transform the response into the format expected by our API
    const hourly = data.hourly;
    const timeArray = hourly.time || [];
    const weatherData = [];

    for (let i = 0; i < timeArray.length; i++) {
      const windDir = hourly.wind_direction_10m?.[i] || 0;
      const windDirRad = (windDir * Math.PI) / 180;
      
      weatherData.push({
        date: timeArray[i],
        temperature_2m: hourly.temperature_2m?.[i] ?? null,
        apparent_temperature: hourly.apparent_temperature?.[i] ?? null,
        relative_humidity_2m: hourly.relative_humidity_2m?.[i] ?? null,
        vapour_pressure_deficit: hourly.vapour_pressure_deficit?.[i] ?? null,
        pressure_msl: hourly.pressure_msl?.[i] ?? null,
        precipitation: hourly.precipitation?.[i] ?? null,
        cloud_cover: hourly.cloud_cover?.[i] ?? null,
        cloud_cover_low: hourly.cloud_cover_low?.[i] ?? null,
        cloud_cover_mid: hourly.cloud_cover_mid?.[i] ?? null,
        cloud_cover_high: hourly.cloud_cover_high?.[i] ?? null,
        et0_fao_evapotranspiration: hourly.et0_fao_evapotranspiration?.[i] ?? null,
        sunshine_duration: hourly.sunshine_duration?.[i] ?? null,
        wind_speed_10m: hourly.wind_speed_10m?.[i] ?? null,
        wind_gusts_10m: hourly.wind_gusts_10m?.[i] ?? null,
        wind_direction_10m: windDir,
        wind_dir_cos_10m: Math.cos(windDirRad),
        wind_dir_sin_10m: Math.sin(windDirRad),
      });
    }

    return weatherData;
  } catch (error) {
    throw new Error(`Failed to fetch weather forecast: ${error.message}`);
  }
};

