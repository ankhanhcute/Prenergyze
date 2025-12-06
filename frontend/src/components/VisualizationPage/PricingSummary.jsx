import React from 'react';

const PricingSummary = ({ metrics }) => {
  if (!metrics) {
    return null;
  }

  const { avgPrice, totalCost, peakPrice, peakPriceTime, totalLoad } = metrics;

  // Format large numbers (e.g. 15000 -> 15k)
  const formatLargeNumber = (num) => {
    if (num >= 1000000) return `${(num / 1000000).toFixed(2)}M`;
    if (num >= 1000) return `${(num / 1000).toFixed(1)}k`;
    return num.toFixed(0);
  };

  return (
    <div className="card" style={{ background: 'linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(6, 78, 59, 0.1) 100%)', border: '1px solid rgba(16, 185, 129, 0.3)' }}>
      <h2 style={{ color: '#34d399', marginTop: 0, marginBottom: '20px', fontSize: '1.2rem', display: 'flex', alignItems: 'center', gap: '10px' }}>
        <span style={{ fontSize: '1.5rem' }}>$</span> Cost & Pricing Simulation
      </h2>
      
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '20px' }}>
        
        {/* Avg Price */}
        <div style={{ background: 'rgba(0, 0, 0, 0.2)', padding: '15px', borderRadius: '8px' }}>
          <div style={{ fontSize: '0.9rem', color: '#a7f3d0', marginBottom: '5px' }}>Avg. Electricity Price</div>
          <div style={{ fontSize: '1.5rem', fontWeight: 'bold', color: '#fff' }}>
            ${avgPrice.toFixed(2)} <span style={{ fontSize: '0.9rem', fontWeight: 'normal', color: '#6ee7b7' }}>/MWh</span>
          </div>
        </div>

        {/* Total Estimated Cost */}
        <div style={{ background: 'rgba(0, 0, 0, 0.2)', padding: '15px', borderRadius: '8px' }}>
          <div style={{ fontSize: '0.9rem', color: '#a7f3d0', marginBottom: '5px' }}>Est. Total Cost</div>
          <div style={{ fontSize: '1.5rem', fontWeight: 'bold', color: '#fff' }}>
            ${formatLargeNumber(totalCost)}
          </div>
          <div style={{ fontSize: '0.8rem', color: '#6ee7b7', marginTop: '5px' }}>
            For {formatLargeNumber(totalLoad)} MWh total demand
          </div>
        </div>

        {/* Peak Price */}
        <div style={{ background: 'rgba(0, 0, 0, 0.2)', padding: '15px', borderRadius: '8px' }}>
          <div style={{ fontSize: '0.9rem', color: '#a7f3d0', marginBottom: '5px' }}>Peak Price</div>
          <div style={{ fontSize: '1.5rem', fontWeight: 'bold', color: '#fff' }}>
            ${peakPrice.toFixed(2)} <span style={{ fontSize: '0.9rem', fontWeight: 'normal', color: '#6ee7b7' }}>/MWh</span>
          </div>
          {peakPriceTime && (
            <div style={{ fontSize: '0.8rem', color: '#6ee7b7', marginTop: '5px' }}>
              at {new Date(peakPriceTime).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
            </div>
          )}
        </div>

      </div>
      
      <div style={{ marginTop: '15px', fontSize: '0.8rem', color: '#6ee7b7', fontStyle: 'italic' }}>
        * Estimated costs based on simulated market pricing model (Load-dependent + Time of Use).
      </div>
    </div>
  );
};

export default PricingSummary;

