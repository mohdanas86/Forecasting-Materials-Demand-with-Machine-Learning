import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
import sys
import warnings
warnings.filterwarnings('ignore')

# Add backend to path
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path))

def build_features(use_database=True, mongodb_url=None, sample_fraction=0.1):
    """
    Build feature engineering pipeline for inventory forecasting

    Args:
        use_database: Whether to load data from MongoDB (True) or CSV files (False)
        mongodb_url: MongoDB connection URL (required if use_database=True)
        sample_fraction: Fraction of data to sample for feature engineering (0.1 = 10%)
    """

    print("🔧 Building features for inventory forecasting...")

    if use_database:
        if not mongodb_url:
            mongodb_url = "mongodb+srv://anas:anas@food.t6wubmw.mongodb.net/inverntory"
        data = load_data_from_database(mongodb_url)
    else:
        data = load_data_from_csv()

    # Merge all datasets
    features_df = merge_datasets(data)

    # Sample the data to reduce memory usage
    if sample_fraction < 1.0:
        print(f"📊 Sampling {sample_fraction*100:.1f}% of data for feature engineering...")
        features_df = features_df.sample(frac=sample_fraction, random_state=42).reset_index(drop=True)
        print(f"✅ Sampled to {len(features_df)} records")

    # Create time series features
    features_df = create_time_series_features(features_df)

    # Create temporal features
    features_df = create_temporal_features(features_df)

    # Create project-related features
    features_df = create_project_features(features_df)

    # Create material-related features
    features_df = create_material_features(features_df)

    # Create external factor features
    features_df = create_external_features(features_df)

    # Clean and prepare final dataset
    features_df = prepare_final_dataset(features_df)

    # Save features
    save_features(features_df)

    print(f"✅ Feature engineering complete! Generated {len(features_df)} feature records")
    print(f"📊 Features saved to: data/features/")

    return features_df

def load_data_from_database(mongodb_url):
    """Load data from MongoDB collections"""
    print("📥 Loading data from MongoDB...")

    async def load_collections():
        client = AsyncIOMotorClient(mongodb_url)
        db = client["inventory_forecasting"]

        try:
            # Load all collections
            collections = {
                'projects': [],
                'materials': [],
                'historical_demand': [],
                'inventory': [],
                'pipeline': [],
                'purchase_orders': []
            }

            for collection_name in collections.keys():
                collection = db[collection_name]
                cursor = collection.find({})
                documents = await cursor.to_list(length=None)
                collections[collection_name] = documents

            return collections

        finally:
            client.close()

    # Run async function
    import asyncio
    data = asyncio.run(load_collections())

    # Convert to DataFrames
    dataframes = {}
    for name, docs in data.items():
        if docs:
            df = pd.DataFrame(docs)
            # Remove MongoDB _id column
            if '_id' in df.columns:
                df = df.drop('_id', axis=1)
            dataframes[name] = df
        else:
            dataframes[name] = pd.DataFrame()

    print(f"✅ Loaded {len(dataframes)} datasets from database")
    return dataframes

def load_data_from_csv():
    """Load data from CSV files"""
    print("📥 Loading data from CSV files...")

    data_dir = Path("data/synthetic")

    dataframes = {
        'projects': pd.read_csv(data_dir / "projects.csv"),
        'materials': pd.read_csv(data_dir / "materials.csv"),
        'historical_demand': pd.read_csv(data_dir / "historical_demand.csv"),
        'inventory': pd.read_csv(data_dir / "inventory.csv"),
        'pipeline': pd.read_csv(data_dir / "pipeline.csv"),
        'purchase_orders': pd.read_csv(data_dir / "purchase_orders.csv")
    }

    print(f"✅ Loaded {len(dataframes)} datasets from CSV")
    return dataframes

def merge_datasets(data):
    """Merge all datasets into a single feature engineering base"""
    print("🔗 Merging datasets...")

    # Start with historical demand as base
    df = data['historical_demand'].copy()

    # Convert date columns to datetime
    df['date'] = pd.to_datetime(df['date'])

    # Merge with projects
    df = df.merge(data['projects'], on='project_id', how='left')

    # Merge with materials
    df = df.merge(data['materials'], on='material_id', how='left')

    # Convert project dates
    df['start_date'] = pd.to_datetime(df['start_date'])
    df['end_date'] = pd.to_datetime(df['end_date'])

    # Add inventory data (most recent inventory for each material)
    if not data['inventory'].empty:
        data['inventory']['date'] = pd.to_datetime(data['inventory']['date'])
        # Get latest inventory per material
        latest_inventory = data['inventory'].sort_values('date').groupby('material_id').last().reset_index()
        df = df.merge(latest_inventory[['material_id', 'stock_level']], on='material_id', how='left')
        df = df.rename(columns={'stock_level': 'current_stock'})

    # Add pipeline data (aggregate planned quantities per project-material)
    if not data['pipeline'].empty:
        data['pipeline']['delivery_date'] = pd.to_datetime(data['pipeline']['delivery_date'])
        # Aggregate pipeline quantities
        pipeline_agg = data['pipeline'].groupby(['project_id', 'material_id'])['planned_qty'].sum().reset_index()
        pipeline_agg = pipeline_agg.rename(columns={'planned_qty': 'pipeline_qty'})
        df = df.merge(pipeline_agg, on=['project_id', 'material_id'], how='left')

    # Add purchase orders data (aggregate incoming quantities per material)
    if not data['purchase_orders'].empty:
        data['purchase_orders']['order_date'] = pd.to_datetime(data['purchase_orders']['order_date'])
        data['purchase_orders']['expected_delivery'] = pd.to_datetime(data['purchase_orders']['expected_delivery'])
        # Aggregate PO quantities expected in next 3 months
        future_cutoff = datetime.now() + timedelta(days=90)
        incoming_po = data['purchase_orders'][
            data['purchase_orders']['expected_delivery'] <= future_cutoff
        ].groupby('material_id')['qty'].sum().reset_index()
        incoming_po = incoming_po.rename(columns={'qty': 'incoming_po_qty'})
        df = df.merge(incoming_po, on='material_id', how='left')

    # Fill missing values
    df['current_stock'] = df['current_stock'].fillna(0)
    df['pipeline_qty'] = df['pipeline_qty'].fillna(0)
    df['incoming_po_qty'] = df['incoming_po_qty'].fillna(0)

    print(f"✅ Merged datasets into {len(df)} records")
    return df

def create_time_series_features(df):
    """Create lag and rolling statistics features"""
    print("📈 Creating time series features...")

    # Sort by project, material, date
    df = df.sort_values(['project_id', 'material_id', 'date']).reset_index(drop=True)

    # Create lag features (1, 3, 6 months)
    for lag in [1, 3, 6]:
        df[f'quantity_lag_{lag}m'] = df.groupby(['project_id', 'material_id'])['quantity'].shift(lag)

    # Create rolling mean features (3, 6 months) using transform
    for window in [3, 6]:
        df[f'quantity_rolling_mean_{window}m'] = df.groupby(['project_id', 'material_id'])['quantity'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )

    # Create rolling std features (3, 6 months) using transform
    for window in [3, 6]:
        df[f'quantity_rolling_std_{window}m'] = df.groupby(['project_id', 'material_id'])['quantity'].transform(
            lambda x: x.rolling(window=window, min_periods=1).std()
        )

    # Fill NaN values in rolling features with 0 (for early periods)
    rolling_cols = [col for col in df.columns if 'rolling' in col]
    df[rolling_cols] = df[rolling_cols].fillna(0)

    print("✅ Created lag and rolling features")
    return df

def create_temporal_features(df):
    """Create temporal features from date columns"""
    print("📅 Creating temporal features...")

    # Extract date components
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['year'] = df['date'].dt.year
    df['day_of_year'] = df['date'].dt.dayofyear

    # Create cyclical features for month
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    print("✅ Created temporal features")
    return df

def create_project_features(df):
    """Create project-related features"""
    print("🏗️ Creating project features...")

    # Days to project start/end from current date
    current_date = datetime.now()
    df['days_to_project_start'] = (df['start_date'] - current_date).dt.days
    df['days_to_project_end'] = (df['end_date'] - current_date).dt.days

    # Project duration in days
    df['project_duration_days'] = (df['end_date'] - df['start_date']).dt.days

    # Project progress (0-1 scale)
    df['project_progress'] = np.clip(
        (current_date - df['start_date']).dt.days / df['project_duration_days'],
        0, 1
    )

    # Project phase categories
    df['project_phase'] = pd.cut(
        df['project_progress'],
        bins=[0, 0.25, 0.5, 0.75, 1.0],
        labels=['planning', 'early', 'mid', 'late']
    )

    # Budget per day
    df['budget_per_day'] = df['budget'] / df['project_duration_days']

    print("✅ Created project features")
    return df

def create_material_features(df):
    """Create material-related features"""
    print("🔧 Creating material features...")

    # Material category based on unit
    unit_to_category = {
        'km': 'linear',
        'MT': 'weight',
        'units': 'count',
        'pieces': 'count'
    }
    df['material_category'] = df['unit'].map(unit_to_category)

    # Lead time in months
    df['lead_time_months'] = df['lead_time_days'] / 30

    # Stock coverage (current stock / average demand)
    # Calculate average demand per material over last 6 months
    recent_demand = df[df['date'] >= (df['date'].max() - timedelta(days=180))]
    avg_demand = recent_demand.groupby('material_id')['quantity'].mean().reset_index()
    avg_demand = avg_demand.rename(columns={'quantity': 'avg_demand_6m'})

    df = df.merge(avg_demand, on='material_id', how='left')
    df['stock_coverage_months'] = df['current_stock'] / df['avg_demand_6m'].fillna(1)
    df['stock_coverage_months'] = df['stock_coverage_months'].fillna(0)

    # Inventory turnover ratio (demand / stock)
    df['inventory_turnover'] = df['avg_demand_6m'] / df['current_stock'].replace(0, 1)

    print("✅ Created material features")
    return df

def create_external_features(df):
    """Create external factor features"""
    print("🌦️ Creating external features...")

    # Monsoon flag (June to September)
    df['monsoon_flag'] = df['month'].isin([6, 7, 8, 9]).astype(int)

    # Holiday flag (simplified - major Indian holidays)
    # Diwali (October/November), Holi (March), Independence Day (August 15), Republic Day (January 26)
    major_holidays = [
        (1, 26),   # Republic Day
        (3, 14),   # Holi (approximate)
        (8, 15),   # Independence Day
        (10, 31),  # Diwali (approximate)
        (11, 1),   # Diwali
    ]

    df['holiday_flag'] = 0
    for month, day in major_holidays:
        df.loc[(df['month'] == month) & (df['date'].dt.day == day), 'holiday_flag'] = 1

    # Seasonal demand multiplier
    seasonal_multipliers = {
        1: 0.8,   # January - Post-winter
        2: 0.7,   # February - Winter
        3: 0.9,   # March - Pre-summer
        4: 1.0,   # April - Summer start
        5: 1.1,   # May - Pre-monsoon
        6: 0.6,   # June - Monsoon start
        7: 0.5,   # July - Heavy monsoon
        8: 0.6,   # August - Monsoon
        9: 0.7,   # September - Monsoon end
        10: 1.2,  # October - Post-monsoon peak
        11: 1.3,  # November - Construction peak
        12: 0.9   # December - Winter slowdown
    }
    df['seasonal_multiplier'] = df['month'].map(seasonal_multipliers)

    # Weather impact factor (higher in monsoon)
    df['weather_impact'] = df['monsoon_flag'] * 0.3  # 30% productivity reduction in monsoon

    print("✅ Created external features")
    return df

def prepare_final_dataset(df):
    """Prepare final dataset for ML"""
    print("🎯 Preparing final dataset...")

    # Select and order columns for ML
    feature_cols = [
        # Target
        'quantity',

        # Identifiers
        'project_id', 'material_id', 'date',

        # Lag features
        'quantity_lag_1m', 'quantity_lag_3m', 'quantity_lag_6m',

        # Rolling features
        'quantity_rolling_mean_3m', 'quantity_rolling_std_3m',
        'quantity_rolling_mean_6m', 'quantity_rolling_std_6m',

        # Temporal features
        'month', 'quarter', 'year', 'day_of_year',
        'month_sin', 'month_cos',

        # Project features
        'days_to_project_start', 'days_to_project_end',
        'project_duration_days', 'project_progress', 'budget_per_day',

        # Material features
        'lead_time_days', 'lead_time_months', 'current_stock',
        'incoming_po_qty', 'pipeline_qty', 'stock_coverage_months',
        'inventory_turnover', 'tax_rate',

        # External features
        'monsoon_flag', 'holiday_flag', 'seasonal_multiplier', 'weather_impact',

        # Categorical features
        'location', 'tower_type', 'substation_type', 'unit', 'material_category', 'project_phase'
    ]

    # Keep only available columns
    available_cols = [col for col in feature_cols if col in df.columns]
    df_final = df[available_cols].copy()

    # Handle categorical columns first (fill with mode or 'unknown')
    categorical_cols = ['location', 'tower_type', 'substation_type', 'unit', 'material_category', 'project_phase']
    for col in categorical_cols:
        if col in df_final.columns:
            # Fill NaN with most frequent value or 'unknown'
            mode_val = df_final[col].mode()
            if not mode_val.empty:
                df_final[col] = df_final[col].fillna(mode_val.iloc[0])
            else:
                df_final[col] = df_final[col].fillna('unknown')

    # Handle any remaining NaN values in numeric columns
    numeric_cols = df_final.select_dtypes(include=[np.number]).columns
    df_final[numeric_cols] = df_final[numeric_cols].fillna(0)

    # Convert categorical columns to category dtype
    for col in categorical_cols:
        if col in df_final.columns:
            df_final[col] = df_final[col].astype('category')

    print(f"✅ Final dataset prepared with {len(df_final)} rows and {len(df_final.columns)} features")
    return df_final

def save_features(df):
    """Save features to CSV format (parquet requires additional dependencies)"""
    print("💾 Saving features...")

    # Create output directory
    output_dir = Path("data/features")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as CSV (human readable and widely supported)
    csv_path = output_dir / "features.csv"
    df.to_csv(csv_path, index=False)
    print(f"✅ Saved features to {csv_path}")

    # Save feature info
    feature_info = {
        'total_samples': len(df),
        'total_features': len(df.columns),
        'feature_types': df.dtypes.to_dict(),
        'categorical_features': [col for col in df.columns if df[col].dtype.name == 'category'],
        'numeric_features': [col for col in df.columns if df[col].dtype.name in ['int64', 'float64']],
        'date_features': [col for col in df.columns if 'date' in col.lower()],
        'target_column': 'quantity'
    }

    import json
    info_path = output_dir / "feature_info.json"
    with open(info_path, 'w') as f:
        json.dump(feature_info, f, indent=2, default=str)
    print(f"✅ Saved feature info to {info_path}")

    # Print feature summary
    print(f"\n📊 Feature Summary:")
    print(f"   Total samples: {len(df)}")
    print(f"   Numeric features: {len(feature_info['numeric_features'])}")
    print(f"   Categorical features: {len(feature_info['categorical_features'])}")
    print(f"   Date features: {len(feature_info['date_features'])}")

if __name__ == "__main__":
    # Build features using database with sampling to reduce memory usage
    features = build_features(use_database=True, sample_fraction=0.1)  # Use 10% sample

    # Print summary
    print("\n" + "="*60)
    print("FEATURE ENGINEERING SUMMARY")
    print("="*60)
    print(f"Total samples: {len(features)}")
    print(f"Total features: {len(features.columns)}")
    print(f"Date range: {features['date'].min()} to {features['date'].max()}")
    print(f"Projects: {features['project_id'].nunique()}")
    print(f"Materials: {features['material_id'].nunique()}")
    print("\nReady for ML model training! 🚀")