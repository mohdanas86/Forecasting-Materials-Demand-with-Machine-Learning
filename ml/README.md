# ML Directory (Machine Learning)

This directory contains all machine learning code for the POWERGRID inventory forecasting system, including data processing, model training, and prediction pipelines.

## 📁 Directory Structure

```
ml/
├── synth_generator.py          # Synthetic data generation
├── data_cleaning.py            # Data preprocessing and cleaning
├── feature_engineering.py      # Feature creation and engineering
├── forecasting_models.py       # Base forecasting model classes
├── training.py                 # Model training orchestration
├── load_synthetic_data.py      # Data loading utilities
├── procurement.py              # Procurement planning logic
├── test_database.py            # Database testing utilities
├── test_synthetic_data.py      # Data validation tests
├── features.py                 # Feature computation functions
├── train_prophet.py            # Prophet model training
├── train_lgb.py                # LightGBM model training
├── ensemble.py                 # Ensemble model implementation
├── models/                     # Trained model artifacts
│   ├── ensemble/              # Ensemble model files
│   │   ├── ensemble_model.pkl
│   │   ├── evaluation_metrics.json
│   │   ├── feature_names.json
│   │   └── model_info.json
│   └── lgb/                   # LightGBM model files
│       ├── lightgbm_model.pkl
│       ├── evaluation_metrics.json
│       ├── feature_names.json
│       ├── model_info.json
│       └── categorical_columns.json
└── outputs/                   # Model predictions and reports
    ├── final_forecast.csv     # Ensemble forecast results
    ├── procurement_plan.csv   # Procurement recommendations
    ├── procurement_report.md  # Procurement analysis
    ├── lgb_predictions.csv    # LightGBM predictions
    └── prophet_forecast.csv   # Prophet predictions
```

## 🤖 ML Pipeline Overview

### 1. Data Generation (`synth_generator.py`)
- Generates realistic POWERGRID project data
- Creates time-series demand patterns
- Produces supplier and material relationships

### 2. Data Processing (`data_cleaning.py`)
- Handles missing values and outliers
- Standardizes data formats
- Validates data integrity

### 3. Feature Engineering (`feature_engineering.py`)
- Creates time-series features (lags, rolling stats)
- Generates seasonal indicators
- Computes project-related features

### 4. Model Training
- **Prophet** (`train_prophet.py`): Facebook's time-series forecasting
- **LightGBM** (`train_lgb.py`): Gradient boosting for regression
- **Ensemble** (`ensemble.py`): Combines multiple models

### 5. Procurement Planning (`procurement.py`)
- Calculates safety stock levels
- Determines reorder points
- Generates procurement recommendations

## 🚀 Quick Start

### Generate Synthetic Data
```bash
cd ml
python synth_generator.py
```

### Train All Models
```bash
# Train individual models
python train_prophet.py
python train_lgb.py

# Create ensemble
python ensemble.py
```

### Generate Forecasts
```bash
python forecasting_models.py
```

### Run Procurement Planning
```bash
python procurement.py
```

## 📊 Models and Algorithms

### Prophet Model
**File**: `train_prophet.py`
**Algorithm**: Facebook Prophet for time-series forecasting
**Features**:
- Handles seasonality and holidays
- Automatic trend detection
- Uncertainty quantification (P10/P50/P90)

**Output**: `outputs/prophet_forecast.csv`

### LightGBM Model
**File**: `train_lgb.py`
**Algorithm**: LightGBM gradient boosting
**Features**:
- Fast training and prediction
- Handles categorical features
- Feature importance analysis

**Output**: `outputs/lgb_predictions.csv`

### Ensemble Model
**File**: `ensemble.py`
**Algorithm**: Weighted ensemble of Prophet and LightGBM
**Features**:
- Combines multiple forecasting approaches
- Uncertainty quantification
- Model validation and metrics

**Output**: `outputs/final_forecast.csv`

## 📈 Model Performance

### Evaluation Metrics
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Square Error)
- **MAPE** (Mean Absolute Percentage Error)
- **R² Score**

### Forecast Horizons
- **Short-term**: 1-4 weeks
- **Medium-term**: 1-3 months
- **Long-term**: 3-12 months

### Uncertainty Quantification
- **P10**: 10th percentile (conservative estimate)
- **P50**: 50th percentile (median forecast)
- **P90**: 90th percentile (optimistic estimate)

## 🛠 Key Scripts

### `synth_generator.py`
**Purpose**: Generate synthetic POWERGRID data
```python
from synth_generator import generate_all_data

# Generate all synthetic data
generate_all_data()
```

**Parameters**:
- `n_projects`: Number of projects (default: 50)
- `n_materials`: Number of materials (default: 100)
- `date_range`: Time period for data generation

### `feature_engineering.py`
**Purpose**: Create ML features from raw data
```python
from feature_engineering import create_all_features

# Create features for all materials
features_df = create_all_features()
```

**Feature Types**:
- **Time Features**: day_of_week, month, quarter
- **Lag Features**: 1-30 day lags
- **Rolling Statistics**: 7, 14, 30-day means
- **Seasonal Features**: monthly/quarterly indicators

### `ensemble.py`
**Purpose**: Train and evaluate ensemble model
```python
from ensemble import train_ensemble_model

# Train ensemble with default parameters
model, metrics = train_ensemble_model()
```

**Ensemble Weights**:
- Prophet: 40%
- LightGBM: 60%

## 📋 Data Requirements

### Input Data Format
- **Historical Demand**: CSV with columns [date, material_id, quantity]
- **Materials**: CSV with material specifications
- **Projects**: CSV with project timelines
- **Inventory**: CSV with current stock levels

### Data Quality
- ✅ No missing dates in time series
- ✅ Consistent material IDs across files
- ✅ Valid date ranges
- ✅ Reasonable quantity values

## 🔧 Configuration

### Model Hyperparameters

#### LightGBM
```python
params = {
    'objective': 'regression',
    'metric': 'mae',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9
}
```

#### Prophet
```python
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    seasonality_mode='multiplicative'
)
```

### Feature Engineering Settings
```python
LAG_PERIODS = [1, 7, 14, 30]
ROLLING_WINDOWS = [7, 14, 30]
SEASONAL_PERIODS = [7, 30, 90]
```

## 📊 Output Files

### `final_forecast.csv`
**Columns**:
- `material_id`: Material identifier
- `date`: Forecast date
- `p10, p50, p90`: Forecast percentiles
- `ensemble_prediction`: Final forecast value

### `procurement_plan.csv`
**Columns**:
- `material_id`: Material identifier
- `current_stock`: Current inventory level
- `safety_stock`: Recommended safety stock
- `reorder_point`: Reorder trigger point
- `recommended_order_qty`: Suggested order quantity
- `recommended_order_date`: When to place order

### Model Artifacts
- **`.pkl` files**: Trained model objects
- **`.json` files**: Model metadata and metrics
- **Feature names**: List of features used in training

## 🧪 Testing and Validation

### Data Validation
```bash
python test_synthetic_data.py
```

### Model Validation
```bash
python -c "from ensemble import validate_ensemble; validate_ensemble()"
```

### Database Testing
```bash
python test_database.py
```

## 📈 Model Retraining

### Automated Retraining
```bash
# Full pipeline
python synth_generator.py && python feature_engineering.py && python ensemble.py
```

### Incremental Updates
```bash
# Update with new data
python load_synthetic_data.py --update
python ensemble.py --retrain
```

## 🔍 Model Interpretability

### Feature Importance
- LightGBM provides built-in feature importance
- Permutation importance analysis
- Partial dependence plots

### Forecast Uncertainty
- Prediction intervals (P10-P90)
- Confidence scores
- Error distribution analysis

## 🚨 Monitoring and Maintenance

### Model Performance Tracking
- Track forecast accuracy over time
- Monitor prediction intervals
- Alert on model degradation

### Data Drift Detection
- Compare new data distributions
- Retrain triggers based on accuracy thresholds
- Automated model updates

## 🐛 Troubleshooting

### Common Issues

#### Memory Errors
```
Solution: Reduce batch sizes in training scripts
```

#### Poor Model Performance
```
Solution: Check feature engineering, adjust hyperparameters
```

#### Data Loading Errors
```
Solution: Validate CSV formats, check file paths
```

#### Import Errors
```
Solution: Install required packages (lightgbm, prophet, scikit-learn)
```

## 📚 Dependencies

### Core ML Libraries
- **lightgbm**: Gradient boosting framework
- **prophet**: Time-series forecasting
- **scikit-learn**: Machine learning utilities
- **pandas**: Data manipulation
- **numpy**: Numerical computing

### Data Processing
- **matplotlib**: Plotting and visualization
- **seaborn**: Statistical visualization
- **plotly**: Interactive charts

### Development
- **jupyter**: Interactive development
- **pytest**: Testing framework
- **black**: Code formatting

## 🔄 Development Workflow

1. **Data Generation**: `python synth_generator.py`
2. **Feature Engineering**: `python feature_engineering.py`
3. **Model Training**: `python train_lgb.py && python ensemble.py`
4. **Validation**: `python test_synthetic_data.py`
5. **Procurement**: `python procurement.py`
6. **Integration**: Update backend API endpoints

## 📋 Contributing

### Code Standards
- Use type hints for function parameters
- Add docstrings to all functions
- Follow PEP 8 style guidelines
- Write unit tests for new functions

### Model Development
- Document hyperparameter choices
- Include model validation metrics
- Provide feature importance analysis
- Test on multiple scenarios

### Documentation
- Update this README for new models
- Document model limitations
- Include performance benchmarks
- Provide usage examples