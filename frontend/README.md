# Prenergyze Frontend

React-based frontend application for the Prenergyze Energy Load Forecasting System.

## Features

- **Entry Page**: Welcome screen with app purpose, research abstract, and model information
- **Visualization Page**: Interactive data visualizations including:
  - Historical load data charts
  - Correlation heatmaps between weather variables and load
  - Weather forecast selector
  - ML model predictions with forecasted weather data

## Setup

### Prerequisites

- Node.js 16+ and npm/yarn
- Backend API running on `http://localhost:8000` (or configure via environment variable)

### Installation

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. (Optional) Configure API base URL by creating a `.env` file:
   ```env
   VITE_API_BASE_URL=http://localhost:8000
   ```

### Development

Start the development server:
```bash
npm run dev
```

The application will be available at `http://localhost:3000`

### Build

Build for production:
```bash
npm run build
```

The built files will be in the `dist` directory.

### Preview Production Build

Preview the production build:
```bash
npm run preview
```

## Project Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── Layout/          # Navigation and layout components
│   │   ├── EntryPage/        # Entry page components
│   │   └── VisualizationPage/ # Visualization components
│   ├── pages/                # Page components
│   ├── services/             # API service layer
│   ├── utils/                # Utility functions
│   └── styles/               # CSS styles
├── public/                   # Static assets
└── package.json             # Dependencies and scripts
```

## Technologies

- **React 18** - UI library
- **React Router** - Routing
- **Recharts** - Charting library
- **Axios** - HTTP client
- **Vite** - Build tool and dev server

## API Integration

The frontend communicates with:
- **Backend API** (`/api/*`) - FastAPI backend for model predictions and historical data
- **Open-Meteo API** - Weather forecast data

## Environment Variables

- `VITE_API_BASE_URL` - Base URL for the FastAPI backend (default: `http://localhost:8000`)
