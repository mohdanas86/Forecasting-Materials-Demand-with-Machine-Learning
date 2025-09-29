#!/usr/bin/env python3
"""
Real Data Ingestion Script for POWERGRID Inventory Management System

This script ingests real public datasets from various sources and maps them
to the project's standardized schema for materials demand forecasting.

Supported data sources:
- OGD India power project data
- Kaggle demand forecasting datasets
- Fuel price data
- Wholesale Price Index (WPI) series
- India shapefiles for geospatial data

Usage:
    python data/ingest_real_data.py --src data/raw --out data/processed
"""

import argparse
import json
import logging
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Optional imports - handle gracefully if not available
try:
    import geopandas as gpd
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False
    gpd = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_ingestion.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress pandas warnings for cleaner output
warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)


class DataIngestionError(Exception):
    """Custom exception for data ingestion errors."""
    pass


class DataValidator:
    """Validates and cleans data according to project schema requirements."""

    @staticmethod
    def validate_units(df: pd.DataFrame, unit_column: str = 'unit') -> pd.DataFrame:
        """Standardize units (kg → MT, etc.) and validate unit consistency."""
        if unit_column not in df.columns:
            logger.warning(f"Unit column '{unit_column}' not found in data")
            return df

        # Unit standardization mapping
        unit_mapping = {
            'kg': 'MT',
            'kilogram': 'MT',
            'tonnes': 'MT',
            'tons': 'MT',
            'tonne': 'MT',
            'mt': 'MT',
            'MT': 'MT',
            'metric ton': 'MT',
            'km': 'km',
            'kilometer': 'km',
            'units': 'units',
            'pieces': 'pieces',
            'piece': 'pieces'
        }

        df[unit_column] = df[unit_column].str.lower().map(unit_mapping).fillna(df[unit_column])
        logger.info(f"Standardized units: {df[unit_column].value_counts().to_dict()}")
        return df

    @staticmethod
    def validate_quantities(df: pd.DataFrame, quantity_columns: List[str]) -> pd.DataFrame:
        """Ensure quantities are non-negative and handle missing values."""
        for col in quantity_columns:
            if col not in df.columns:
                continue

            # Convert to numeric, coerce errors to NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')

            # Check for negative values
            negative_count = (df[col] < 0).sum()
            if negative_count > 0:
                logger.warning(f"Found {negative_count} negative values in {col}, setting to 0")
                df.loc[df[col] < 0, col] = 0

            # Fill missing values with 0 for quantities
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                logger.warning(f"Found {missing_count} missing values in {col}, filling with 0")
                df[col] = df[col].fillna(0)

        return df

    @staticmethod
    def validate_dates(df: pd.DataFrame, date_columns: List[str]) -> pd.DataFrame:
        """Validate and standardize date columns."""
        for col in date_columns:
            if col not in df.columns:
                continue

            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                invalid_dates = df[col].isna().sum()
                if invalid_dates > 0:
                    logger.warning(f"Found {invalid_dates} invalid dates in {col}")
            except Exception as e:
                logger.error(f"Error parsing dates in {col}: {e}")

        return df

    @staticmethod
    def remove_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
        """Remove duplicate rows based on specified columns."""
        initial_count = len(df)
        df = df.drop_duplicates(subset=subset, keep='first')
        removed_count = initial_count - len(df)

        if removed_count > 0:
            logger.info(f"Removed {removed_count} duplicate rows")
        else:
            logger.info("No duplicate rows found")

        return df


class DataMapper:
    """Maps external data sources to project schema."""

    # Schema definitions
    PROJECTS_SCHEMA = {
        'project_id': 'str',
        'name': 'str',
        'state': 'str',
        'lat': 'float64',
        'lon': 'float64',
        'start_date': 'datetime64[ns]',
        'end_date': 'datetime64[ns]',
        'budget': 'float64',
        'tower_type': 'str',
        'substation_type': 'str'
    }

    MATERIALS_SCHEMA = {
        'material_id': 'str',
        'name': 'str',
        'unit': 'str',
        'tax_rate': 'float64',
        'base_price': 'float64'
    }

    HISTORICAL_DEMAND_SCHEMA = {
        'project_id': 'str',
        'material_id': 'str',
        'date': 'datetime64[ns]',
        'quantity': 'float64'
    }

    INVENTORY_SCHEMA = {
        'warehouse_id': 'str',
        'material_id': 'str',
        'date': 'datetime64[ns]',
        'stock_level': 'float64'
    }

    @staticmethod
    def map_ogd_projects(df: pd.DataFrame) -> pd.DataFrame:
        """Map OGD India power project data to projects schema."""
        logger.info("Mapping OGD projects data...")

        # Expected OGD columns (adjust based on actual data)
        column_mapping = {
            'project_id': ['project_id', 'id', 'project_code'],
            'name': ['project_name', 'name', 'title'],
            'state': ['state', 'location', 'region'],
            'start_date': ['start_date', 'commencement_date', 'planned_start'],
            'end_date': ['end_date', 'completion_date', 'planned_end'],
            'budget': ['budget', 'cost', 'estimated_cost'],
            'tower_type': ['tower_type', 'transmission_type'],
            'substation_type': ['substation_type', 'substation']
        }

        mapped_df = DataMapper._map_columns(df, column_mapping)

        # Add default values for missing required columns
        if 'lat' not in mapped_df.columns:
            mapped_df['lat'] = None
        if 'lon' not in mapped_df.columns:
            mapped_df['lon'] = None

        # Ensure project_id is unique and string
        if 'project_id' in mapped_df.columns:
            mapped_df['project_id'] = mapped_df['project_id'].astype(str).str.zfill(3)

        return mapped_df

    @staticmethod
    def map_kaggle_materials(df: pd.DataFrame) -> pd.DataFrame:
        """Map Kaggle materials data to materials schema."""
        logger.info("Mapping Kaggle materials data...")

        column_mapping = {
            'material_id': ['material_id', 'id', 'item_id', 'product_id'],
            'name': ['name', 'material_name', 'item_name', 'description'],
            'unit': ['unit', 'unit_of_measure', 'uom'],
            'tax_rate': ['tax_rate', 'gst_rate', 'vat_rate'],
            'base_price': ['base_price', 'price', 'unit_price', 'cost']
        }

        mapped_df = DataMapper._map_columns(df, column_mapping)

        # Set defaults for missing columns
        if 'tax_rate' not in mapped_df.columns:
            mapped_df['tax_rate'] = 0.18  # Default GST rate
        if 'base_price' not in mapped_df.columns:
            mapped_df['base_price'] = 0.0

        # Ensure material_id is properly formatted
        if 'material_id' in mapped_df.columns:
            mapped_df['material_id'] = mapped_df['material_id'].astype(str).str.zfill(3)

        return mapped_df

    @staticmethod
    def map_demand_data(df: pd.DataFrame) -> pd.DataFrame:
        """Map demand forecasting data to historical_demand schema."""
        logger.info("Mapping demand data...")

        column_mapping = {
            'project_id': ['project_id', 'project', 'site_id'],
            'material_id': ['material_id', 'material', 'item_id'],
            'date': ['date', 'timestamp', 'period'],
            'quantity': ['quantity', 'demand', 'qty', 'amount']
        }

        mapped_df = DataMapper._map_columns(df, column_mapping)

        # Ensure IDs are properly formatted
        for col in ['project_id', 'material_id']:
            if col in mapped_df.columns:
                mapped_df[col] = mapped_df[col].astype(str).str.zfill(3)

        return mapped_df

    @staticmethod
    def map_inventory_data(df: pd.DataFrame) -> pd.DataFrame:
        """Map inventory data to inventory schema."""
        logger.info("Mapping inventory data...")

        column_mapping = {
            'warehouse_id': ['warehouse_id', 'warehouse', 'location_id', 'site'],
            'material_id': ['material_id', 'material', 'item_id'],
            'date': ['date', 'timestamp'],
            'stock_level': ['stock_level', 'inventory', 'quantity', 'qty']
        }

        mapped_df = DataMapper._map_columns(df, column_mapping)

        # Set default warehouse if missing
        if 'warehouse_id' not in mapped_df.columns:
            mapped_df['warehouse_id'] = 'WH001'

        # Ensure material_id is properly formatted
        if 'material_id' in mapped_df.columns:
            mapped_df['material_id'] = mapped_df['material_id'].astype(str).str.zfill(3)

        return mapped_df

    @staticmethod
    def _map_columns(df: pd.DataFrame, mapping: Dict[str, List[str]]) -> pd.DataFrame:
        """Generic column mapping utility."""
        result_df = pd.DataFrame()

        for target_col, source_cols in mapping.items():
            # Find the first available source column
            for source_col in source_cols:
                if source_col in df.columns:
                    result_df[target_col] = df[source_col]
                    logger.debug(f"Mapped {source_col} -> {target_col}")
                    break
            else:
                logger.warning(f"No source column found for {target_col} from {source_cols}")

        return result_df


class DataLoader:
    """Handles loading data from various sources and formats."""

    def __init__(self, raw_dir: Path):
        self.raw_dir = raw_dir
        self.session = self._create_session()

    def _create_session(self) -> requests.Session:
        """Create a requests session with retry strategy."""
        session = requests.Session()
        retry = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session

    def load_csv_files(self) -> Dict[str, pd.DataFrame]:
        """Load all CSV files from raw directory."""
        csv_files = {}
        for csv_file in self.raw_dir.glob('*.csv'):
            try:
                logger.info(f"Loading CSV: {csv_file.name}")
                df = pd.read_csv(csv_file)
                csv_files[csv_file.stem] = df
                logger.info(f"Loaded {len(df)} rows from {csv_file.name}")
            except Exception as e:
                logger.error(f"Failed to load {csv_file.name}: {e}")

        return csv_files

    def load_zip_files(self) -> Dict[str, pd.DataFrame]:
        """Load CSV files from ZIP archives."""
        zip_files = {}
        for zip_path in self.raw_dir.glob('*.zip'):
            try:
                logger.info(f"Processing ZIP: {zip_path.name}")
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    # Find CSV files in the ZIP
                    csv_names = [name for name in zf.namelist() if name.endswith('.csv')]
                    for csv_name in csv_names:
                        with zf.open(csv_name) as f:
                            df = pd.read_csv(f)
                            key = f"{zip_path.stem}_{Path(csv_name).stem}"
                            zip_files[key] = df
                            logger.info(f"Loaded {len(df)} rows from {csv_name} in {zip_path.name}")
            except Exception as e:
                logger.error(f"Failed to process {zip_path.name}: {e}")

        return zip_files

    def load_shapefiles(self) -> Optional[gpd.GeoDataFrame]:
        """Load shapefiles for geospatial data."""
        if not HAS_GEOPANDAS:
            logger.warning("GeoPandas not available, skipping shapefile loading")
            return None

        shapefile_paths = list(self.raw_dir.glob('*.shp'))
        if not shapefile_paths:
            logger.info("No shapefiles found")
            return None

        try:
            # Load the first shapefile found
            shp_path = shapefile_paths[0]
            logger.info(f"Loading shapefile: {shp_path.name}")
            gdf = gpd.read_file(shp_path)
            logger.info(f"Loaded {len(gdf)} geospatial features")
            return gdf
        except Exception as e:
            logger.error(f"Failed to load shapefile: {e}")
            return None

    def download_file(self, url: str, filename: str) -> Optional[Path]:
        """Download a file from URL to raw directory."""
        try:
            logger.info(f"Downloading {filename} from {url}")
            response = self.session.get(url, timeout=30)
            response.raise_for_status()

            file_path = self.raw_dir / filename
            with open(file_path, 'wb') as f:
                f.write(response.content)

            logger.info(f"Downloaded {filename} ({len(response.content)} bytes)")
            return file_path
        except Exception as e:
            logger.error(f"Failed to download {filename}: {e}")
            return None


class DataIngestionPipeline:
    """Main pipeline for data ingestion and processing."""

    def __init__(self, raw_dir: Path, processed_dir: Path):
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.loader = DataLoader(raw_dir)
        self.mapper = DataMapper()
        self.validator = DataValidator()
        self.report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'sources_processed': 0,
            'total_records': 0,
            'output_files': [],
            'anomalies': [],
            'missing_fields': {}
        }

    def run_ingestion(self) -> Dict[str, Any]:
        """Run the complete data ingestion pipeline."""
        logger.info("Starting data ingestion pipeline...")

        try:
            # Load all available data
            csv_data = self.loader.load_csv_files()
            zip_data = self.loader.load_zip_files()
            shapefile_data = self.loader.load_shapefiles()

            all_data = {**csv_data, **zip_data}
            self.report['sources_processed'] = len(all_data)

            # Process each data source
            processed_data = {}
            for source_name, df in all_data.items():
                logger.info(f"Processing source: {source_name}")
                processed_df = self._process_data_source(source_name, df)
                if processed_df is not None:
                    processed_data[source_name] = processed_df

            # Merge and consolidate data
            consolidated_data = self._consolidate_data(processed_data, shapefile_data)

            # Save processed data
            self._save_processed_data(consolidated_data)

            # Generate report
            self._generate_report()

            logger.info("Data ingestion pipeline completed successfully")
            return self.report

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            self.report['error'] = str(e)
            return self.report

    def _process_data_source(self, source_name: str, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Process a single data source."""
        try:
            # Detect data type and apply appropriate mapping
            if self._is_projects_data(source_name, df):
                processed_df = self.mapper.map_ogd_projects(df)
                processed_df = self._validate_projects_data(processed_df)
            elif self._is_materials_data(source_name, df):
                processed_df = self.mapper.map_kaggle_materials(df)
                processed_df = self._validate_materials_data(processed_df)
            elif self._is_demand_data(source_name, df):
                processed_df = self.mapper.map_demand_data(df)
                processed_df = self._validate_demand_data(processed_df)
            elif self._is_inventory_data(source_name, df):
                processed_df = self.mapper.map_inventory_data(df)
                processed_df = self._validate_inventory_data(processed_df)
            else:
                logger.warning(f"Could not determine data type for {source_name}")
                return None

            self.report['total_records'] += len(processed_df)
            return processed_df

        except Exception as e:
            logger.error(f"Failed to process {source_name}: {e}")
            self.report['anomalies'].append({
                'source': source_name,
                'error': str(e)
            })
            return None

    def _is_projects_data(self, source_name: str, df: pd.DataFrame) -> bool:
        """Detect if data contains project information."""
        project_indicators = ['project', 'site', 'location', 'budget', 'tower', 'substation']
        return any(indicator in source_name.lower() for indicator in project_indicators) or \
               any(col.lower() in ['project_id', 'budget', 'start_date'] for col in df.columns)

    def _is_materials_data(self, source_name: str, df: pd.DataFrame) -> bool:
        """Detect if data contains material information."""
        material_indicators = ['material', 'item', 'product', 'inventory']
        return any(indicator in source_name.lower() for indicator in material_indicators) or \
               any(col.lower() in ['material_id', 'unit', 'price'] for col in df.columns)

    def _is_demand_data(self, source_name: str, df: pd.DataFrame) -> bool:
        """Detect if data contains demand information."""
        demand_indicators = ['demand', 'forecast', 'historical', 'consumption']
        return any(indicator in source_name.lower() for indicator in demand_indicators) or \
               any(col.lower() in ['quantity', 'demand', 'date'] for col in df.columns)

    def _is_inventory_data(self, source_name: str, df: pd.DataFrame) -> bool:
        """Detect if data contains inventory information."""
        inventory_indicators = ['inventory', 'stock', 'warehouse']
        return any(indicator in source_name.lower() for indicator in inventory_indicators) or \
               any(col.lower() in ['stock_level', 'warehouse_id'] for col in df.columns)

    def _validate_projects_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate projects data."""
        df = self.validator.validate_dates(df, ['start_date', 'end_date'])
        df = self.validator.validate_quantities(df, ['budget', 'lat', 'lon'])
        df = self.validator.remove_duplicates(subset=['project_id'])
        return df

    def _validate_materials_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate materials data."""
        df = self.validator.validate_units(df, 'unit')
        df = self.validator.validate_quantities(df, ['tax_rate', 'base_price'])
        df = self.validator.remove_duplicates(subset=['material_id'])
        return df

    def _validate_demand_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate demand data."""
        df = self.validator.validate_dates(df, ['date'])
        df = self.validator.validate_quantities(df, ['quantity'])
        df = self.validator.remove_duplicates(subset=['project_id', 'material_id', 'date'])
        return df

    def _validate_inventory_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate inventory data."""
        df = self.validator.validate_dates(df, ['date'])
        df = self.validator.validate_quantities(df, ['stock_level'])
        df = self.validator.remove_duplicates(subset=['warehouse_id', 'material_id', 'date'])
        return df

    def _consolidate_data(self, processed_data: Dict[str, pd.DataFrame],
                         shapefile_data: Optional[gpd.GeoDataFrame]) -> Dict[str, pd.DataFrame]:
        """Consolidate data from multiple sources."""
        logger.info("Consolidating data from multiple sources...")

        consolidated = {
            'projects': pd.DataFrame(),
            'materials': pd.DataFrame(),
            'historical_demand': pd.DataFrame(),
            'inventory': pd.DataFrame()
        }

        # Group processed data by type
        for source_name, df in processed_data.items():
            if self._is_projects_data(source_name, df):
                consolidated['projects'] = pd.concat([consolidated['projects'], df], ignore_index=True)
            elif self._is_materials_data(source_name, df):
                consolidated['materials'] = pd.concat([consolidated['materials'], df], ignore_index=True)
            elif self._is_demand_data(source_name, df):
                consolidated['historical_demand'] = pd.concat([consolidated['historical_demand'], df], ignore_index=True)
            elif self._is_inventory_data(source_name, df):
                consolidated['inventory'] = pd.concat([consolidated['inventory'], df], ignore_index=True)

        # Apply final validation and deduplication
        for data_type, df in consolidated.items():
            if not df.empty:
                if data_type == 'projects':
                    consolidated[data_type] = self._validate_projects_data(df)
                elif data_type == 'materials':
                    consolidated[data_type] = self._validate_materials_data(df)
                elif data_type == 'historical_demand':
                    consolidated[data_type] = self._validate_demand_data(df)
                elif data_type == 'inventory':
                    consolidated[data_type] = self._validate_inventory_data(df)

                logger.info(f"Consolidated {len(df)} records for {data_type}")

        # Add geospatial data if available
        if shapefile_data is not None and not consolidated['projects'].empty:
            consolidated['projects'] = self._join_geospatial_data(consolidated['projects'], shapefile_data)

        return consolidated

    def _join_geospatial_data(self, projects_df: pd.DataFrame, gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        """Join geospatial data with projects."""
        if not HAS_GEOPANDAS:
            return projects_df

        try:
            logger.info("Joining geospatial data with projects...")

            # This is a simplified geospatial join - in practice, you'd need proper
            # coordinate matching or state-based joins
            if 'state' in projects_df.columns and 'state' in gdf.columns:
                # Merge on state name
                projects_gdf = gpd.GeoDataFrame(projects_df)
                merged = projects_gdf.merge(gdf, on='state', how='left')
                logger.info("Successfully joined geospatial data")
                return pd.DataFrame(merged)
            else:
                logger.warning("Cannot join geospatial data - missing state columns")
                return projects_df

        except Exception as e:
            logger.error(f"Failed to join geospatial data: {e}")
            return projects_df

    def _save_processed_data(self, consolidated_data: Dict[str, pd.DataFrame]) -> None:
        """Save processed data to CSV files."""
        logger.info("Saving processed data...")

        for data_type, df in consolidated_data.items():
            if df.empty:
                logger.warning(f"No data to save for {data_type}")
                continue

            output_path = self.processed_dir / f"{data_type}.csv"
            try:
                df.to_csv(output_path, index=False)
                logger.info(f"Saved {len(df)} records to {output_path}")
                self.report['output_files'].append({
                    'file': str(output_path),
                    'records': len(df),
                    'columns': list(df.columns)
                })
            except Exception as e:
                logger.error(f"Failed to save {data_type}: {e}")

    def _generate_report(self) -> None:
        """Generate JSON report of the ingestion process."""
        report_path = self.processed_dir / "ingest_report.json"
        try:
            with open(report_path, 'w') as f:
                json.dump(self.report, f, indent=2, default=str)
            logger.info(f"Generated ingestion report: {report_path}")
        except Exception as e:
            logger.error(f"Failed to generate report: {e}")


def main():
    """Main entry point for the data ingestion script."""
    parser = argparse.ArgumentParser(
        description="Ingest real public datasets for POWERGRID inventory management"
    )
    parser.add_argument(
        '--src',
        type=str,
        default='data/raw',
        help='Source directory containing raw data files'
    )
    parser.add_argument(
        '--out',
        type=str,
        default='data/processed',
        help='Output directory for processed data'
    )
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Set logging level'
    )

    args = parser.parse_args()

    # Configure logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Convert to Path objects
    raw_dir = Path(args.src)
    processed_dir = Path(args.out)

    # Ensure directories exist
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Check for required dependencies
    if not HAS_GEOPANDAS:
        logger.warning("GeoPandas not available - shapefile processing will be skipped")
        logger.warning("Install with: pip install geopandas")

    # Run ingestion pipeline
    pipeline = DataIngestionPipeline(raw_dir, processed_dir)
    report = pipeline.run_ingestion()

    # Print summary
    print("\n" + "="*60)
    print("DATA INGESTION SUMMARY")
    print("="*60)
    print(f"Sources processed: {report['sources_processed']}")
    print(f"Total records: {report['total_records']}")
    print(f"Output files: {len(report['output_files'])}")
    print(f"Anomalies: {len(report['anomalies'])}")

    if report['output_files']:
        print("\nOutput files:")
        for output in report['output_files']:
            print(f"  - {output['file']}: {output['records']} records")

    if report.get('error'):
        print(f"\n❌ Error: {report['error']}")
        return 1

    print("\n✅ Data ingestion completed successfully!")
    return 0


if __name__ == '__main__':
    exit(main())