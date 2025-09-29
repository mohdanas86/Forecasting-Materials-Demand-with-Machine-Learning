import pandas as pd
import numpy as np
from prophet import Prophet
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load features data and prepare for Prophet modeling"""
    print("📥 Loading feature data...")

    # Load from synthetic data directly (not sampled features)
    synthetic_path = Path("data/synthetic/historical_demand.csv")
    if synthetic_path.exists():
        print("📊 Loading from synthetic historical demand data...")
        df = pd.read_csv(synthetic_path)
        print(f"✅ Loaded {len(df)} historical demand records")

        # Aggregate to monthly and add regressors
        monthly_data = prepare_monthly_from_synthetic(df)
    else:
        # Fallback to features
        features_path = Path("data/features/features.csv")
        df = pd.read_csv(features_path)
        print(f"✅ Loaded {len(df)} feature records")
        monthly_data = prepare_monthly_data(df)

    return monthly_data

def prepare_monthly_from_synthetic(df):
    """Aggregate synthetic historical demand to monthly level for Prophet"""
    print("📊 Aggregating synthetic data to monthly demand...")

    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])

    # Group by material_id and month, aggregate demand
    monthly_df = df.groupby(['material_id', pd.Grouper(key='date', freq='M')]).agg({
        'quantity': 'sum'  # Total monthly demand
    }).reset_index()

    # Rename columns for Prophet
    monthly_df = monthly_df.rename(columns={
        'date': 'ds',
        'quantity': 'y'
    })

    # Add basic regressors (simplified for synthetic data)
    monthly_df['holiday_flag'] = 0  # Default no holidays
    monthly_df['pipeline_qty'] = monthly_df['y'] * 0.1  # Assume 10% pipeline
    monthly_df['seasonal_multiplier'] = 1.0  # Default
    monthly_df['weather_impact'] = 0.0  # Default

    # Set monsoon flag based on month (June-September)
    monthly_df['monsoon_flag'] = monthly_df['ds'].dt.month.isin([6, 7, 8, 9]).astype(int)

    # Set holiday flag for major holidays
    major_holidays = [
        (1, 26),   # Republic Day
        (3, 14),   # Holi (approximate)
        (8, 15),   # Independence Day
        (10, 31),  # Diwali (approximate)
    ]
    monthly_df['holiday_flag'] = 0
    for month, day in major_holidays:
        monthly_df.loc[
            (monthly_df['ds'].dt.month == month) &
            (monthly_df['ds'].dt.day == day),
            'holiday_flag'
        ] = 1

    return monthly_df

def prepare_monthly_data(df):
    """Aggregate daily data to monthly level for Prophet"""
    print("📊 Aggregating to monthly demand...")

    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])

    # Group by material_id and month, aggregate demand and regressors
    monthly_df = df.groupby(['material_id', pd.Grouper(key='date', freq='M')]).agg({
        'quantity': 'sum',  # Total monthly demand
        'holiday_flag': 'max',  # Any holiday in the month
        'pipeline_qty': 'mean',  # Average pipeline quantity
        'monsoon_flag': lambda x: x.mean() > 0.5,  # Was monsoon active most of the month?
        'seasonal_multiplier': 'mean',
        'weather_impact': 'mean'
    }).reset_index()

    # Rename columns for Prophet
    monthly_df = monthly_df.rename(columns={
        'date': 'ds',
        'quantity': 'y'
    })

    # Convert boolean columns to int
    monthly_df['holiday_flag'] = monthly_df['holiday_flag'].astype(int)
    monthly_df['monsoon_flag'] = monthly_df['monsoon_flag'].astype(int)

    print(f"✅ Prepared {len(monthly_df)} monthly records for {monthly_df['material_id'].nunique()} materials")

    return monthly_df

def train_prophet_model(material_data, material_id):
    """Train Prophet model for a specific material"""
    print(f"🔮 Training Prophet model for material {material_id}...")

    # Prepare data for this material
    mat_df = material_data.copy()

    # Initialize Prophet model
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,  # No weekly pattern in monthly data
        daily_seasonality=False,   # No daily pattern in monthly data
        seasonality_mode='multiplicative',
        interval_width=0.8  # 80% confidence interval
    )

    # Add regressors
    model.add_regressor('holiday_flag')
    model.add_regressor('pipeline_qty')
    # model.add_regressor('monsoon_flag')  # Removed - using for conditional seasonality
    model.add_regressor('seasonal_multiplier')
    model.add_regressor('weather_impact')

    # Add custom seasonality for monsoon period (June-September)
    model.add_seasonality(
        name='monsoon_season',
        period=12,  # Annual
        fourier_order=3,
        condition_name='monsoon_flag'
    )

    try:
        # Fit the model
        model.fit(mat_df)

        print(f"✅ Trained model for material {material_id}")
        return model

    except Exception as e:
        print(f"❌ Failed to train model for material {material_id}: {e}")
        return None

def generate_forecast(model, material_data, material_id, forecast_periods=6):
    """Generate forecast for next N months"""
    print(f"🔮 Generating {forecast_periods}-month forecast for material {material_id}...")

    if model is None:
        return None

    # Get the last date from training data
    last_date = material_data['ds'].max()

    # Create future dataframe
    future_dates = pd.date_range(
        start=last_date + pd.offsets.MonthBegin(1),
        periods=forecast_periods,
        freq='M'
    )

    # Create future dataframe with regressor values
    # Use average values from recent history for regressors
    recent_data = material_data.tail(3)  # Last 3 months

    future_df = pd.DataFrame({
        'ds': future_dates,
        'holiday_flag': recent_data['holiday_flag'].mean().round().astype(int),
        'pipeline_qty': recent_data['pipeline_qty'].mean(),
        # 'monsoon_flag': 0,  # Removed - using for conditional seasonality
        'seasonal_multiplier': recent_data['seasonal_multiplier'].mean(),
        'weather_impact': 0  # Assume no weather impact
    })

    # Set monsoon flag based on month (June-September) - for conditional seasonality
    future_df['monsoon_flag'] = future_df['ds'].dt.month.isin([6, 7, 8, 9]).astype(int)

    try:
        # Generate forecast
        forecast = model.predict(future_df)

        # Extract prediction intervals
        forecast_result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        forecast_result['material_id'] = material_id

        # Calculate percentiles (approximate p10, p50, p90)
        # Using normal distribution assumption for prediction intervals
        forecast_result['p50'] = forecast_result['yhat']  # Median prediction
        forecast_result['p10'] = forecast_result['yhat_lower']  # Approximate 10th percentile
        forecast_result['p90'] = forecast_result['yhat_upper']  # Approximate 90th percentile

        # Ensure non-negative predictions
        forecast_result[['p10', 'p50', 'p90']] = forecast_result[['p10', 'p50', 'p90']].clip(lower=0)

        print(f"✅ Generated forecast for material {material_id}")
        return forecast_result[['material_id', 'ds', 'p10', 'p50', 'p90']]

    except Exception as e:
        print(f"❌ Failed to generate forecast for material {material_id}: {e}")
        return None

def save_forecasts(all_forecasts):
    """Save all forecasts to CSV"""
    print("💾 Saving forecasts...")

    if not all_forecasts:
        print("❌ No forecasts to save")
        return

    # Combine all forecasts
    combined_forecasts = pd.concat(all_forecasts, ignore_index=True)

    # Rename columns
    combined_forecasts = combined_forecasts.rename(columns={'ds': 'date'})

    # Sort by material_id and date
    combined_forecasts = combined_forecasts.sort_values(['material_id', 'date'])

    # Create output directory
    output_dir = Path("ml/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    output_path = output_dir / "prophet_forecast.csv"
    combined_forecasts.to_csv(output_path, index=False)

    print(f"✅ Saved {len(combined_forecasts)} forecast records to {output_path}")

    # Print summary
    print("\n📊 Forecast Summary:")
    print(f"   Materials forecasted: {combined_forecasts['material_id'].nunique()}")
    print(f"   Forecast periods: {len(combined_forecasts) // combined_forecasts['material_id'].nunique()}")
    print(f"   Date range: {combined_forecasts['date'].min()} to {combined_forecasts['date'].max()}")

def main():
    """Main function to train Prophet models and generate forecasts"""
    print("🚀 Starting Prophet Forecasting Pipeline")
    print("=" * 50)

    try:
        # Load and prepare data
        monthly_data = load_and_prepare_data()

        # Get unique materials
        materials = monthly_data['material_id'].unique()
        print(f"📋 Found {len(materials)} materials to forecast")

        # Train models and generate forecasts
        all_forecasts = []

        for material_id in materials:
            print(f"\n🔄 Processing material {material_id}...")

            # Get data for this material
            material_data = monthly_data[monthly_data['material_id'] == material_id].copy()

            # Skip if insufficient data
            if len(material_data) < 6:  # Need at least 6 months of data
                print(f"⚠️  Skipping material {material_id}: insufficient data ({len(material_data)} months)")
                continue

            # Train model
            model = train_prophet_model(material_data, material_id)

            # Generate forecast
            if model is not None:
                forecast = generate_forecast(model, material_data, material_id, forecast_periods=6)
                if forecast is not None:
                    all_forecasts.append(forecast)

        # Save all forecasts
        save_forecasts(all_forecasts)

        print("\n" + "=" * 50)
        print("✅ Prophet forecasting pipeline completed!")
        print("=" * 50)

    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()