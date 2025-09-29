# Tests Directory

This directory contains test suites for validating the POWERGRID inventory forecasting system, including backend API tests, frontend component tests, and data validation tests.

## 📁 Directory Structure

```
tests/
├── test_backend.py        # Backend API and functionality tests
└── test_frontend.js       # Frontend component and integration tests
```

## 🧪 Testing Overview

### Test Categories

#### Backend Tests (`test_backend.py`)
- **API Endpoint Testing**: Validate all FastAPI routes
- **Data Processing**: Test data loading and validation
- **ML Model Integration**: Verify model loading and predictions
- **Database Operations**: Test MongoDB Atlas connectivity

#### Frontend Tests (`test_frontend.js`)
- **Component Testing**: React component rendering and interactions
- **API Integration**: Axios request/response handling
- **UI Functionality**: Form submissions, data display, error handling
- **Routing**: React Router navigation and route protection

## 🚀 Running Tests

### Backend Tests
```bash
cd tests
python test_backend.py
```

### Frontend Tests
```bash
cd tests
npm test test_frontend.js
```

### All Tests (with Docker)
```bash
cd docker
docker-compose exec backend python /app/tests/test_backend.py
docker-compose exec frontend npm test /app/tests/test_frontend.js
```

## 📋 Test Coverage

### Backend Test Suite

#### API Endpoint Tests
```python
def test_forecast_endpoints():
    # Test /forecast with various parameters
    # Test /forecast/summary
    # Validate response schemas

def test_alerts_endpoints():
    # Test /alerts with filtering
    # Test /alerts/summary
    # Test /alerts/types

def test_upload_endpoints():
    # Test file upload validation
    # Test data processing
    # Test database insertion
```

#### Data Validation Tests
```python
def test_data_loading():
    # Test synthetic data loading
    # Validate data schemas
    # Check data integrity

def test_ml_model_loading():
    # Test model artifact loading
    # Validate prediction outputs
    # Check model performance
```

#### Integration Tests
```python
def test_full_pipeline():
    # Test data → features → predictions → alerts
    # Validate end-to-end functionality
    # Performance benchmarking
```

### Frontend Test Suite

#### Component Tests
```javascript
describe('UploadData Component', () => {
  it('renders file upload forms', () => {
    // Test form rendering
    // Test file input handling
  });

  it('handles file uploads', () => {
    // Test upload API calls
    // Test success/error states
  });
});
```

#### Integration Tests
```javascript
describe('API Integration', () => {
  it('fetches forecast data', () => {
    // Test API service calls
    // Test data transformation
    // Test error handling
  });
});
```

## 🛠 Test Setup

### Backend Test Dependencies
```python
# requirements-test.txt
pytest==7.4.0
pytest-asyncio==0.21.1
httpx==0.24.1
pytest-mock==3.11.1
```

### Frontend Test Dependencies
```json
{
  "devDependencies": {
    "@testing-library/react": "^13.4.0",
    "@testing-library/jest-dom": "^5.16.5",
    "@testing-library/user-event": "^14.4.3",
    "jest": "^27.5.1",
    "jest-environment-jsdom": "^29.5.0"
  }
}
```

## 📊 Test Data

### Mock Data
- **Synthetic API Responses**: Pre-recorded API responses for testing
- **Sample Files**: Test CSV/JSON files for upload testing
- **Mock Models**: Lightweight model artifacts for testing

### Test Fixtures
```python
@pytest.fixture
def sample_forecast_data():
    return {
        "forecasts": [...],
        "metadata": {...}
    }

@pytest.fixture
def mock_api_response():
    return Mock(status_code=200, json=lambda: {...})
```

## 🔄 Test Execution

### Continuous Integration
```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]
jobs:
  test-backend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Test Backend
        run: python tests/test_backend.py

  test-frontend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Test Frontend
        run: npm test tests/test_frontend.js
```

### Local Development Testing
```bash
# Run tests on file changes
pytest-watch tests/

# Run with coverage
pytest --cov=backend --cov-report=html tests/

# Run specific test
pytest tests/test_backend.py::test_forecast_endpoints
```

## 📈 Test Metrics

### Coverage Targets
- **Backend**: >90% code coverage
- **Frontend**: >80% component coverage
- **Integration**: All critical user journeys

### Performance Benchmarks
- **API Response Time**: <500ms for forecast endpoints
- **File Upload**: <30 seconds for 1MB files
- **ML Prediction**: <2 seconds per material

## 🐛 Debugging Tests

### Common Issues

#### Backend Test Failures
```
Error: Connection refused on port 8000
Solution: Start backend server before running tests
```

#### Frontend Test Failures
```
Error: Cannot find module 'axios'
Solution: Install test dependencies
```

#### Async Test Timeouts
```
Error: Test timeout exceeded
Solution: Increase timeout or mock slow operations
```

### Debug Mode
```python
# Enable debug logging in tests
import logging
logging.basicConfig(level=logging.DEBUG)

# Run tests with verbose output
pytest -v -s tests/
```

## 📝 Writing New Tests

### Backend Test Template
```python
import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_new_endpoint():
    response = client.get("/new-endpoint")
    assert response.status_code == 200
    assert "expected_key" in response.json()
```

### Frontend Test Template
```javascript
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import MyComponent from '../src/components/MyComponent';

describe('MyComponent', () => {
  it('renders correctly', () => {
    render(<MyComponent />);
    expect(screen.getByText('Expected Text')).toBeInTheDocument();
  });
});
```

## 🔧 Test Configuration

### pytest.ini
```ini
[tool:pytest.ini_options]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --tb=short --strict-markers
markers =
    unit: Unit tests
    integration: Integration tests
    slow: Slow running tests
```

### Jest Configuration
```javascript
// jest.config.js
module.exports = {
  testEnvironment: 'jsdom',
  setupFilesAfterEnv: ['<rootDir>/src/setupTests.js'],
  moduleNameMapping: {
    '\\.(css|less|scss|sass)$': 'identity-obj-proxy',
  },
};
```

## 📊 Test Reporting

### Coverage Reports
```bash
# Generate HTML coverage report
pytest --cov=backend --cov-report=html
open htmlcov/index.html
```

### Test Results
```bash
# JUnit XML output for CI
pytest --junitxml=test-results.xml tests/

# Allure reports
pytest --alluredir=allure-results tests/
allure serve allure-results
```

## 🚨 Test Maintenance

### Updating Tests
- Update test data when schemas change
- Modify assertions when API responses change
- Add new test cases for new features

### Test Data Management
- Keep test data in version control
- Use factories for complex test data
- Avoid hard-coded test data

### CI/CD Integration
- Run tests on every push
- Block merges on test failures
- Monitor test performance trends

## 📋 Best Practices

### Test Organization
- Group related tests in classes
- Use descriptive test names
- Follow AAA pattern (Arrange, Act, Assert)

### Test Isolation
- Each test should be independent
- Use fixtures for setup/teardown
- Mock external dependencies

### Test Readability
- Use clear assertions
- Add comments for complex logic
- Keep tests focused on one behavior

## 🔗 Related Documentation

- **API Documentation**: `http://localhost:8000/docs`
- **Frontend Guide**: `frontend/README.md`
- **Backend Guide**: `backend/README.md`
- **ML Documentation**: `ml/README.md`

## 🤝 Contributing

### Adding New Tests
1. Create test file in `tests/` directory
2. Follow naming convention: `test_*.py` or `test_*.js`
3. Add appropriate test markers
4. Update this README if needed

### Test Standards
- All new features must have tests
- Tests must pass before merging
- Maintain or improve coverage metrics
- Document complex test scenarios