import pandas as pd
from datetime import datetime, timedelta
import os

def clean_historical_demand_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean historical demand data with the following steps:
    1. Remove duplicates (project_id + material_id + date)
    2. Enforce positive quantities
    3. Standardize units (kg → MT)
    4. Fill missing dates with 0 demand

    Args:
        df: DataFrame with columns: project_id, material_id, date, quantity

    Returns:
        Cleaned DataFrame
    """
    print(f"Original data shape: {df.shape}")

    # Convert date column to datetime if it's not already
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])

    # 1. Remove duplicates based on project_id + material_id + date
    df = df.drop_duplicates(subset=['project_id', 'material_id', 'date'])
    print(f"After removing duplicates: {df.shape}")

    # 2. Enforce positive quantities
    negative_count = (df['quantity'] < 0).sum()
    if negative_count > 0:
        print(f"Found {negative_count} negative quantities, setting to 0")
        df.loc[df['quantity'] < 0, 'quantity'] = 0

    # 3. Standardize units (kg → MT)
    # Assuming there's a 'unit' column or we need to infer from quantity values
    if 'unit' in df.columns:
        # Convert kg to MT (1 MT = 1000 kg)
        kg_mask = df['unit'].str.lower() == 'kg'
        if kg_mask.any():
            print(f"Converting {kg_mask.sum()} kg quantities to MT")
            df.loc[kg_mask, 'quantity'] = df.loc[kg_mask, 'quantity'] / 1000
            df.loc[kg_mask, 'unit'] = 'MT'

    # 4. Fill missing dates with 0 demand
    if not df.empty:
        df = _fill_missing_dates(df)

    print(f"Final cleaned data shape: {df.shape}")
    return df

def _fill_missing_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing dates with 0 demand for each project_id + material_id combination
    """
    # Get all unique project-material combinations
    combinations = df[['project_id', 'material_id']].drop_duplicates()

    filled_dfs = []

    for _, combo in combinations.iterrows():
        project_id = combo['project_id']
        material_id = combo['material_id']

        # Filter data for this combination
        combo_df = df[(df['project_id'] == project_id) & (df['material_id'] == material_id)].copy()

        if combo_df.empty:
            continue

        # Get date range
        min_date = combo_df['date'].min()
        max_date = combo_df['date'].max()

        # Create complete date range (daily)
        date_range = pd.date_range(start=min_date, end=max_date, freq='D')

        # Create DataFrame with all dates
        complete_df = pd.DataFrame({
            'date': date_range,
            'project_id': project_id,
            'material_id': material_id,
            'quantity': 0.0
        })

        # Merge with existing data
        merged_df = pd.merge(complete_df, combo_df[['date', 'quantity']],
                           on='date', how='left', suffixes=('', '_existing'))

        # Use existing quantity if available, otherwise 0
        merged_df['quantity'] = merged_df['quantity_existing'].fillna(0)
        merged_df = merged_df.drop('quantity_existing', axis=1)

        # Add unit column if it exists in original
        if 'unit' in combo_df.columns:
            merged_df['unit'] = combo_df['unit'].iloc[0] if not combo_df.empty else 'MT'

        filled_dfs.append(merged_df)

    # Combine all combinations
    result_df = pd.concat(filled_dfs, ignore_index=True) if filled_dfs else pd.DataFrame()

    return result_df

def load_and_clean_data(file_path: str = None, mongo_collection=None) -> pd.DataFrame:
    """
    Load data from CSV file or MongoDB collection and clean it

    Args:
        file_path: Path to CSV file
        mongo_collection: Motor collection object

    Returns:
        Cleaned DataFrame
    """
    if file_path:
        # Load from CSV
        df = pd.read_csv(file_path)
        print(f"Loaded data from {file_path}")
    elif mongo_collection:
        # Load from MongoDB
        cursor = mongo_collection.find({})
        data = list(cursor)
        df = pd.DataFrame(data)
        # Convert ObjectId to string if present
        if '_id' in df.columns:
            df = df.drop('_id', axis=1)
        print(f"Loaded data from MongoDB collection")
    else:
        raise ValueError("Either file_path or mongo_collection must be provided")

    # Clean the data
    cleaned_df = clean_historical_demand_data(df)

    return cleaned_df

def save_cleaned_data(df: pd.DataFrame, output_path: str = None) -> str:
    """
    Save cleaned DataFrame to CSV file

    Args:
        df: Cleaned DataFrame
        output_path: Output file path (optional)

    Returns:
        Path to saved file
    """
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"cleaned_historical_demand_{timestamp}.csv"

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

    return output_path

# Example usage
if __name__ == "__main__":
    # Example with CSV file
    try:
        # Test with sample data
        sample_file = "../sample_historical_demand.csv"
        if os.path.exists(sample_file):
            print("Testing data cleaning with sample file...")
            cleaned_df = load_and_clean_data(sample_file)
            output_file = save_cleaned_data(cleaned_df, "../cleaned_historical_demand.csv")

            print("\nSample of cleaned data:")
            print(cleaned_df.head(10))
        else:
            print("Sample file not found. Use load_and_clean_data() to clean your historical demand data")

    except Exception as e:
        print(f"Error: {e}")