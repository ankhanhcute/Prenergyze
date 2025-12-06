/**
 * Calculate estimated electricity price based on load and time of day.
 * 
 * @param {number} load - Forecasted load in MW
 * @param {Date|string} date - Date/time of the forecast
 * @returns {number} Estimated price in $/MWh
 */
export const calculateElectricityPrice = (load, date) => {
  const dateObj = new Date(date);
  const hour = dateObj.getHours();
  
  // Base parameters
  const BASE_PRICE = 30; // $/MWh
  const GRID_CAPACITY = 50000; // MW (assumed capacity)
  const SLOPE_FACTOR = 100; // Price multiplier for high load
  
  // Load Factor: Price increases exponentially as load approaches capacity
  // Formula: (Load / Capacity)^2 * Slope
  const loadRatio = Math.min(load / GRID_CAPACITY, 1.0);
  const loadComponent = Math.pow(loadRatio, 2.5) * SLOPE_FACTOR;
  
  // Time of Use (TOU) Adjustment
  // Peak hours: 4 PM - 9 PM (16:00 - 21:00) -> Higher price
  // Off-peak: 11 PM - 6 AM -> Lower price
  let timeAdjustment = 0;
  
  if (hour >= 16 && hour <= 21) {
    timeAdjustment = 15; // +$15 during peak
  } else if (hour >= 23 || hour <= 6) {
    timeAdjustment = -5; // -$5 during off-peak
  } else if (hour >= 7 && hour <= 15) {
    timeAdjustment = 5; // +$5 during day
  }
  
  // Seasonality adjustment (Summer/Winter peaks)
  // Simple implementation: Summer (Jun-Sep) is more expensive
  const month = dateObj.getMonth(); // 0-11
  let seasonalMultiplier = 1.0;
  
  if (month >= 5 && month <= 8) { // Jun-Sep
    seasonalMultiplier = 1.2;
  } else if (month === 0 || month === 1 || month === 11) { // Dec-Feb
    seasonalMultiplier = 1.1;
  }
  
  // Final Calculation
  // Price = (Base + LoadComponent + TOU) * Seasonal
  let price = (BASE_PRICE + loadComponent + timeAdjustment) * seasonalMultiplier;
  
  // Ensure logical minimum price
  return Math.max(price, 10.0);
};

/**
 * Calculate summary metrics for a set of forecast data
 * 
 * @param {Array} forecastPoints - Array of objects with { date, load }
 * @returns {Object} Summary metrics { avgPrice, totalCost, peakPrice, peakPriceTime }
 */
export const calculatePricingSummary = (forecastPoints) => {
  if (!forecastPoints || forecastPoints.length === 0) {
    return null;
  }
  
  let totalCost = 0;
  let totalLoad = 0;
  let peakPrice = 0;
  let peakPriceTime = null;
  
  forecastPoints.forEach(point => {
    const price = calculateElectricityPrice(point.load, point.date);
    
    // Cost for this hour = Price ($/MWh) * Load (MW) * 1 hour
    totalCost += price * point.load;
    totalLoad += point.load;
    
    if (price > peakPrice) {
      peakPrice = price;
      peakPriceTime = point.date;
    }
  });
  
  const avgPrice = totalLoad > 0 ? totalCost / totalLoad : 0;
  
  return {
    avgPrice,
    totalCost,
    peakPrice,
    peakPriceTime,
    totalLoad
  };
};

