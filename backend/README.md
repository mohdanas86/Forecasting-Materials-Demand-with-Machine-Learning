# Backend Directory (FastAPI)

This directory contains the FastAPI backend for the POWERGRID inventory forecasting system, providing REST API endpoints for data upload, forecast retrieval, and alert monitoring.

## 📁 Directory Structure

```
backend/
├── main.py                    # FastAPI application entry point
├── auth.py                    # JWT authentication (stub implementation)
├── models.py                  # Pydantic data models
├── requirements.txt           # Python dependencies
├── test_routes.py            # API endpoint testing script
├── routes/                   # API route handlers
│   ├── forecast.py           # Forecast data endpoints
│   ├── alerts.py             # Alert monitoring endpoints
│   └── upload.py             # Data upload endpoints
├── services/                 # Business logic services
│   └── alerts.py             # Alert detection service
└── ml/                       # ML model access (symlinked)
```

## 🚀 Quick Start

### Local Development
```bash
cd backend

# Install dependencies
pip install -r requirements.txt

# Start the server
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### With Docker
```bash
cd docker
docker-compose up backend
```

## 📡 API Endpoints

### Base URL
- **Development**: `http://localhost:8000`
- **API Docs**: `http://localhost:8000/docs`
- **Alternative Docs**: `http://localhost:8000/redoc`

### Forecast Endpoints (`/forecast`)

#### GET `/forecast`
Retrieve forecast data with procurement recommendations.

**Query Parameters:**
- `project_id` (optional): Filter by project ID
- `material_id` (optional): Filter by material ID
- `period` (optional): Time period (`week`, `month`, `quarter`, `all`)

**Response:**
```json
{
  "forecasts": [
    {
      "material_id": "MAT001",
      "forecast_date": "2025-01-15",
      "p10": 85.5,
      "p50": 120.0,
      "p90": 165.2,
      "safety_stock": 25.0,
      "reorder_point": 30.0,
      "recommendations": [...]
    }
  ],
  "metadata": {
    "total_materials": 50,
    "generated_at": "2025-01-15T10:30:00Z"
  }
}
```

#### GET `/forecast/summary`
Get forecast summary statistics.

### Alerts Endpoints (`/alerts`)

#### GET `/alerts`
Retrieve inventory alerts with filtering.

**Query Parameters:**
- `project_id` (optional): Filter by project ID
- `material_id` (optional): Filter by material ID

**Response:**
```json
[
  {
    "alert_id": "shortage_MAT001_20250115",
    "alert_type": "shortage",
    "severity": "high",
    "material_id": "MAT001",
    "title": "Stock Shortage Alert",
    "description": "Forecasted demand exceeds available stock",
    "recommended_action": "Place emergency order for 50 units",
    "created_at": "2025-01-15T10:30:00Z"
  }
]
```

#### GET `/alerts/summary`
Get alert summary statistics.

#### GET `/alerts/types`
Get information about available alert types.

### Upload Endpoints (`/upload`)

#### POST `/upload/projects`
Upload project data (CSV/JSON).

#### POST `/upload/materials`
Upload materials master data.

#### POST `/upload/historical`
Upload historical demand data with optional cleaning.

#### POST `/upload/inventory`
Upload current inventory levels.

**File Format Support:**
- CSV files (with headers)
- JSON files (array of objects)

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
MONGODB_URL=mongodb+srv://username:password@cluster.mongodb.net/database
DEBUG=true
```

### CORS Configuration

The backend is configured with permissive CORS settings for development:
- **Allowed Origins**: All origins (`*`)
- **Allowed Methods**: All HTTP methods
- **Allowed Headers**: All headers

### Database Connection

- **Database**: MongoDB Atlas (cloud)
- **Collections**:
  - `projects` - Project information
  - `materials` - Materials master data
  - `historical_demand` - Historical demand records
  - `inventory` - Current inventory levels

## 🏗 Architecture

### Application Structure

#### `main.py`
- FastAPI application initialization
- CORS middleware setup
- MongoDB connection
- Route registration

#### `routes/`
- **forecast.py**: Forecast data retrieval and processing
- **alerts.py**: Alert generation and filtering
- **upload.py**: File upload handling and validation

#### `services/`
- **alerts.py**: Alert detection logic using forecast vs inventory data

#### `auth.py`
- JWT authentication stub (ready for production implementation)

### Data Flow

1. **Data Upload** → Validation → MongoDB storage
2. **Forecast Request** → ML model loading → Prediction generation
3. **Alert Generation** → Forecast vs inventory comparison → Alert creation

## 📊 Dependencies

### Core Dependencies
- **fastapi**: Modern web framework
- **uvicorn**: ASGI server
- **motor**: Async MongoDB driver
- **pydantic**: Data validation
- **pandas**: Data processing
- **python-multipart**: File upload handling

### Development Dependencies
- **pytest**: Testing framework
- **black**: Code formatting
- **isort**: Import sorting

## 🧪 Testing

### Run API Tests
```bash
cd backend
python test_routes.py
```

### Manual Testing
```bash
# Test root endpoint
curl http://localhost:8000/

# Test forecast endpoint
curl "http://localhost:8000/forecast?period=week"

# Test alerts endpoint
curl http://localhost:8000/alerts
```

## 🔒 Security

### Current Implementation
- **CORS**: Permissive for development
- **Authentication**: JWT stub (not enforced)
- **Input Validation**: Pydantic models
- **File Upload**: Basic validation

### Production Considerations
- Restrict CORS origins
- Implement proper JWT authentication
- Add rate limiting
- Input sanitization
- File type validation

## 📈 Monitoring

### Health Checks
- **GET `/`**: Basic health check
- MongoDB connection status
- ML model loading status

### Logging
- Request/response logging
- Error tracking
- Performance metrics

## 🚀 Deployment

### Docker Deployment
```bash
cd docker
docker-compose up --build
```

### Production Server
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Environment Setup
1. Set production environment variables
2. Configure MongoDB Atlas connection
3. Set up proper CORS origins
4. Enable authentication
5. Configure logging

## 🐛 Troubleshooting

### Common Issues

#### MongoDB Connection Failed
```
Error: MongoDB connection timeout
```
**Solution**: Check MongoDB Atlas connection string and network access.

#### CORS Errors
```
Access to XMLHttpRequest blocked by CORS policy
```
**Solution**: Verify CORS configuration in `main.py`.

#### Import Errors
```
ModuleNotFoundError: No module named 'fastapi'
```
**Solution**: Install dependencies with `pip install -r requirements.txt`.

#### Port Already in Use
```
[Errno 48] Address already in use
```
**Solution**: Kill existing process or use different port.

### Debug Mode
Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📝 API Documentation

### Interactive Documentation
Visit `http://localhost:8000/docs` for:
- Interactive API testing
- Request/response schemas
- Authentication details

### Alternative Documentation
Visit `http://localhost:8000/redoc` for:
- Clean, readable documentation
- Printable format
- Offline viewing

## 🔄 Development Workflow

1. **Code Changes**: Edit files in `backend/`
2. **Testing**: Run `python test_routes.py`
3. **API Testing**: Use `/docs` interface
4. **Database**: Check MongoDB Atlas dashboard
5. **Logs**: Monitor console output

## 📋 Contributing

### Code Style
- Use Black for code formatting
- Use isort for import sorting
- Follow PEP 8 guidelines
- Add type hints where possible

### Testing
- Write unit tests for new functions
- Test API endpoints manually
- Validate data models
- Check error handling

### Documentation
- Update this README for new features
- Document API changes
- Add docstrings to functions
- Update OpenAPI schemas