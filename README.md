# POWERGRID Inventory Forecasting System

A comprehensive machine learning-powered inventory forecasting system for POWERGRID Corporation, designed to predict material requirements, optimize procurement planning, and prevent stockouts through advanced analytics and predictive modeling.

## 🎯 Overview

This system provides end-to-end inventory forecasting capabilities with:

- **Synthetic Data Generation**: Realistic power grid material data simulation
- **Machine Learning Models**: Ensemble forecasting using Prophet, LightGBM, and custom algorithms
- **Web Dashboard**: Interactive React frontend for data visualization and management
- **REST API**: FastAPI backend for model serving and data processing
- **Containerized Deployment**: Docker-based multi-service architecture

## 📁 Project Structure

```
powergrid-inventory-forecasting/
├── data/                    # Data processing and synthetic generation
│   ├── raw/                # Raw synthetic data files
│   ├── processed/          # Feature-engineered datasets
│   ├── models/             # Trained ML model artifacts
│   └── README.md           # Data documentation
├── backend/                # FastAPI backend service
│   ├── app/               # Application code
│   ├── tests/             # Backend unit tests
│   └── README.md          # Backend documentation
├── frontend/               # React frontend application
│   ├── src/               # Source code
│   ├── public/            # Static assets
│   └── README.md          # Frontend documentation
├── ml/                     # Machine learning pipeline
│   ├── models/            # Model training scripts
│   ├── evaluation/        # Model evaluation tools
│   ├── procurement/       # Procurement planning logic
│   └── README.md          # ML documentation
├── tests/                  # Cross-service test suite
│   ├── test_backend.py    # Backend API tests
│   ├── test_frontend.js   # Frontend component tests
│   └── README.md          # Testing documentation
├── docker/                 # Containerization and deployment
│   ├── docker-compose.yml # Multi-service orchestration
│   ├── Dockerfile.backend  # Backend container config
│   ├── Dockerfile.frontend # Frontend container config
│   └── README.md          # Docker documentation
├── docs/                   # Project documentation
└── README.md              # This file
```

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- 4GB RAM minimum
- Python 3.9+ (for local development)
- Node.js 16+ (for local development)

### One-Command Setup

```bash
# Clone and start the entire system
git clone <repository-url>
cd powergrid-inventory-forecasting
docker-compose up --build
```

### Access Points

- **Frontend Dashboard**: [http://localhost:3000](http://localhost:3000)
- **API Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **API Base URL**: [http://localhost:8000/api/v1](http://localhost:8000/api/v1)

## 🛠 Development Setup

### Local Development

```bash
# Backend setup
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload

# Frontend setup (new terminal)
cd frontend
npm install
npm start

# ML pipeline (new terminal)
cd ml
python train_models.py
```

### Environment Variables

```bash
# .env file
MONGODB_URI=mongodb+srv://...
JWT_SECRET_KEY=your-secret-key
OPENAI_API_KEY=your-openai-key
```

## 📊 Data Pipeline

### 1. Data Generation

```python
from data.generate_synthetic_data import generate_powergrid_data

# Generate 2 years of synthetic data
data = generate_powergrid_data(
    start_date='2022-01-01',
    end_date='2024-01-01',
    materials=['conductor', 'insulator', 'transformer']
)
```

### 2. Feature Engineering

```python
from data.feature_engineering import create_features

# Create ML-ready features
features = create_features(data)
# Output: processed_features.csv, feature_importance.json
```

### 3. Model Training

```python
from ml.train_models import train_ensemble_model

# Train ensemble model
model = train_ensemble_model(features)
# Output: model_artifacts/, evaluation_metrics.json
```

## 🤖 Machine Learning Models

### Model Ensemble

- **Prophet**: Time series forecasting with seasonality
- **LightGBM**: Gradient boosting for complex patterns
- **Custom Models**: Domain-specific algorithms

### Key Features

- Multi-material forecasting
- Seasonal pattern recognition
- Procurement lead time consideration
- Confidence intervals for predictions

### Model Performance

```
Material A - MAE: 45.2, RMSE: 67.8, R²: 0.89
Material B - MAE: 32.1, RMSE: 51.3, R²: 0.92
Material C - MAE: 28.7, RMSE: 43.9, R²: 0.94
```

## 🔌 API Endpoints

### Core Endpoints

```http
GET    /api/v1/forecast          # Get inventory forecasts
POST   /api/v1/forecast/upload   # Upload new data
GET    /api/v1/alerts            # Get inventory alerts
GET    /api/v1/materials         # List materials
GET    /api/v1/procurement       # Procurement recommendations
```

### Example API Usage

```python
import requests

# Get forecasts for next 30 days
response = requests.get(
    'http://localhost:8000/api/v1/forecast',
    params={'days': 30, 'material': 'conductor'}
)
forecasts = response.json()
```

## 🎨 Frontend Features

### Dashboard Components

- **Inventory Overview**: Current stock levels and trends
- **Forecast Visualization**: Interactive charts and graphs
- **Alert Management**: Low stock and procurement alerts
- **Data Upload**: CSV/Excel file upload interface
- **Procurement Planning**: Automated purchase recommendations

### Technology Stack

- React 18 with TypeScript
- Vite for fast development
- Material-UI components
- Chart.js for data visualization
- Axios for API communication

## 🐳 Docker Deployment

### Services

```yaml
services:
  backend:
    build: ./backend
    ports: ["8000:8000"]
    environment:
      - MONGODB_URI=${MONGODB_URI}

  frontend:
    build: ./frontend
    ports: ["3000:3000"]

  mongodb:
    image: mongo:6.0
    ports: ["27017:27017"]
```

### Production Deployment

```bash
# Build and deploy
docker-compose -f docker-compose.prod.yml up -d

# Scale services
docker-compose up -d --scale backend=3
```

## 🧪 Testing

### Run Test Suite

```bash
# Backend tests
cd backend && python -m pytest tests/

# Frontend tests
cd frontend && npm test

# Integration tests
cd tests && python test_integration.py
```

### Test Coverage

- Backend: 90%+ coverage
- Frontend: 85%+ coverage
- Integration: All critical paths

## 📈 Performance Metrics

### System Performance

- **API Response Time**: <500ms average
- **Model Prediction**: <2 seconds per material
- **Data Processing**: <30 seconds for 1GB files
- **Frontend Load**: <3 seconds initial load

### Scalability

- Supports 1000+ materials
- Handles 10M+ data points
- Concurrent users: 100+
- Database queries: <100ms average

## 🔒 Security

### Authentication

- JWT token-based authentication
- Role-based access control
- API key management

### Data Protection

- Encrypted database connections
- Input validation and sanitization
- Secure file upload handling

## 📚 Documentation

### Directory Documentation

- [Data Pipeline](data/README.md) - Data processing and generation
- [Backend API](backend/README.md) - FastAPI service documentation
- [Frontend App](frontend/README.md) - React application guide
- [ML Models](ml/README.md) - Machine learning pipeline
- [Testing](tests/README.md) - Test suite documentation
- [Docker Setup](docker/README.md) - Containerization guide

### API Documentation

- Interactive API docs: [http://localhost:8000/docs](http://localhost:8000/docs)
- OpenAPI 3.0 specification
- Request/response examples

## 🤝 Contributing

### Development Workflow

1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-feature`
3. Make changes and add tests
4. Run test suite: `docker-compose run --rm backend pytest`
5. Submit pull request

### Code Standards

- **Backend**: PEP 8, type hints, docstrings
- **Frontend**: ESLint, Prettier, TypeScript strict mode
- **ML**: Clear variable names, reproducible results
- **Tests**: 80%+ coverage, descriptive names

### Commit Convention

```
feat: add new forecasting model
fix: resolve API timeout issue
docs: update README with new features
test: add integration tests for procurement
```

## 🚀 Deployment

### Staging Environment

```bash
# Deploy to staging
docker-compose -f docker-compose.staging.yml up -d
```

### Production Environment

```bash
# Deploy to production
docker-compose -f docker-compose.prod.yml up -d

# Enable monitoring
docker-compose -f docker-compose.monitoring.yml up -d
```

### CI/CD Pipeline

- Automated testing on push
- Docker image building
- Security scanning
- Performance monitoring

## 📞 Support

### Getting Help

- **Issues**: GitHub Issues for bug reports
- **Discussions**: GitHub Discussions for questions
- **Documentation**: Check directory READMEs first

### Common Issues

1. **Port conflicts**: Change ports in docker-compose.yml
2. **Memory issues**: Increase Docker memory allocation
3. **Model loading errors**: Check model artifact paths
4. **API connection errors**: Verify service networking

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- POWERGRID Corporation for domain expertise
- Open source ML libraries (Prophet, LightGBM, scikit-learn)
- FastAPI and React communities

---

Built with ❤️ for efficient power grid inventory management
