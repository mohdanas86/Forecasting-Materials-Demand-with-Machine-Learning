# POWERGRID Inventory Forecasting - Frontend

A modern React frontend for the POWERGRID inventory forecasting system built with Vite and Tailwind CSS.

## Features

- **Upload Data**: File upload interface for projects, materials, historical demand, and inventory data
- **Forecast Dashboard**: Interactive charts showing forecast predictions vs historical data using Recharts
- **Alerts**: Real-time inventory alerts with risk assessment and recommendations
- **Responsive Design**: Mobile-friendly interface with Tailwind CSS
- **API Integration**: Axios-based API calls to FastAPI backend

## Tech Stack

- **React 18** - UI framework
- **Vite** - Build tool and development server
- **Tailwind CSS** - Utility-first CSS framework
- **React Router** - Client-side routing
- **Axios** - HTTP client for API calls
- **Recharts** - Chart library for data visualization

## Project Structure

```
src/
├── components/
│   └── Navigation.jsx          # Main navigation component
├── pages/
│   ├── UploadData.jsx          # Data upload page
│   ├── ForecastDashboard.jsx   # Forecast visualization page
│   └── Alerts.jsx              # Alerts monitoring page
├── services/
│   └── api.js                  # API service with axios configuration
├── App.jsx                     # Main app component with routing
├── main.jsx                    # App entry point
└── index.css                   # Global styles
```

## Getting Started

1. **Install dependencies:**

   ```bash
   npm install
   ```

2. **Start development server:**

   ```bash
   npm run dev
   ```

3. **Build for production:**

   ```bash
   npm run build
   ```

4. **Preview production build:**
   ```bash
   npm run preview
   ```

## Docker Development

The project includes Docker support for easy development setup.

### Using Docker Compose

1. **Start all services:**

   ```bash
   cd docker
   docker-compose up --build
   ```

2. **Access the applications:**
   - Frontend: `http://localhost:5173`
   - Backend API: `http://localhost:8000`
   - API Documentation: `http://localhost:8000/docs`

3. **Stop services:**

   ```bash
   docker-compose down
   ```

### Environment Variables

Create a `.env` file in the root directory for MongoDB Atlas connection:

```env
MONGODB_URL=mongodb+srv://your-username:your-password@your-cluster.mongodb.net/your-database
```

## API Integration

The frontend communicates with the FastAPI backend running on `http://localhost:8000`. Make sure the backend is running before using the frontend.

### Available Endpoints

- **Upload Data**: POST `/upload/projects`, `/upload/materials`, `/upload/historical`, `/upload/inventory`
- **Forecast**: GET `/forecast`, GET `/forecast/summary`
- **Alerts**: GET `/alerts`, GET `/alerts/summary`, GET `/alerts/types`

## Pages Overview

### Upload Data Page

- File upload forms for CSV/JSON files
- Real-time upload status and error handling
- Support for projects, materials, historical demand, and inventory data

### Forecast Dashboard

- Interactive charts showing historical vs forecast data
- Procurement recommendations and safety stock levels
- Filterable by material and time period
- Summary statistics and key metrics

### Alerts Page

- Real-time inventory alerts with severity levels
- Filterable by project, material, and severity
- Risk assessment and recommended actions
- Alert type information and guidelines

## Development

- Uses ESLint for code linting
- Hot module replacement during development
- Responsive design with Tailwind CSS
- Component-based architecture

## Backend Integration

Ensure the FastAPI backend is running on port 8000. The frontend includes:

- JWT authentication stub (ready for production auth)
- Error handling for API failures
- Loading states and user feedback
- Data validation and type safety
