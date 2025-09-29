import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os
from pathlib import Path

# Constants for realistic data generation
INDIAN_STATES = [
    'Maharashtra', 'Gujarat', 'Rajasthan', 'Madhya Pradesh', 'Karnataka',
    'Tamil Nadu', 'Andhra Pradesh', 'Telangana', 'Uttar Pradesh', 'Bihar',
    'West Bengal', 'Odisha', 'Chhattisgarh', 'Jharkhand', 'Punjab',
    'Haryana', 'Kerala', 'Assam', 'Delhi', 'Goa', 'Himachal Pradesh',
    'Jammu and Kashmir', 'Uttarakhand', 'Puducherry', 'Meghalaya',
    'Mizoram', 'Nagaland', 'Tripura', 'Sikkim', 'Arunachal Pradesh'
]

TOWER_TYPES = ['HVDC', 'HVAC']  # High Voltage Direct Current, High Voltage Alternating Current
SUBSTATION_TYPES = ['AIS', 'GIS']  # Air Insulated Switchgear, Gas Insulated Switchgear

MATERIALS = [
    {'name': 'ACSR Conductor', 'unit': 'km', 'base_price': 150000, 'tax_rate': 0.18},
    {'name': 'Insulator', 'unit': 'pieces', 'base_price': 2500, 'tax_rate': 0.18},
    {'name': 'Transformer 132kV', 'unit': 'units', 'base_price': 2500000, 'tax_rate': 0.18},
    {'name': 'Circuit Breaker', 'unit': 'units', 'base_price': 750000, 'tax_rate': 0.18},
    {'name': 'Power Cable XLPE', 'unit': 'km', 'base_price': 200000, 'tax_rate': 0.18},
    {'name': 'Steel Tower', 'unit': 'MT', 'base_price': 150000, 'tax_rate': 0.18},
    {'name': 'Foundation Bolts', 'unit': 'MT', 'base_price': 80000, 'tax_rate': 0.18},
    {'name': 'Control Cable', 'unit': 'km', 'base_price': 25000, 'tax_rate': 0.18},
    {'name': 'Lightning Arrester', 'unit': 'units', 'base_price': 150000, 'tax_rate': 0.18},
    {'name': 'CT PT Unit', 'unit': 'units', 'base_price': 300000, 'tax_rate': 0.18}
]

LEAD_TIME_DAYS = [30, 45, 60, 90, 120, 180]  # Realistic lead times

class PowerGridDataGenerator:
    def __init__(self, n_projects=50, m_materials=10, seed=42):
        np.random.seed(seed)
        random.seed(seed)
        self.n_projects = n_projects
        self.m_materials = m_materials
        self.start_date = datetime(2022, 1, 1)
        self.end_date = datetime(2024, 12, 31)

        # Create output directory
        self.output_dir = Path("data/synthetic")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_projects(self):
        """Generate project data with enhanced location and complexity factors"""
        projects = []

        # Location-based budget multipliers (cost of land, labor, infrastructure)
        location_budget_multipliers = {
            'Maharashtra': 1.3, 'Gujarat': 1.2, 'Rajasthan': 0.9, 'Madhya Pradesh': 0.8,
            'Karnataka': 1.1, 'Tamil Nadu': 0.95, 'Andhra Pradesh': 1.0, 'Telangana': 1.05,
            'Uttar Pradesh': 0.85, 'Bihar': 0.75, 'West Bengal': 0.9, 'Odisha': 0.8,
            'Chhattisgarh': 0.7, 'Jharkhand': 0.65, 'Punjab': 0.8, 'Haryana': 0.85,
            'Kerala': 0.9, 'Assam': 0.6, 'Delhi': 1.4, 'Goa': 0.5,
            'Himachal Pradesh': 0.7, 'Jammu and Kashmir': 0.6, 'Uttarakhand': 0.75,
            'Puducherry': 0.8, 'Meghalaya': 0.65, 'Mizoram': 0.6, 'Nagaland': 0.55,
            'Tripura': 0.6, 'Sikkim': 0.7, 'Arunachal Pradesh': 0.5
        }

        # Project complexity multipliers based on tower and substation types
        complexity_multipliers = {
            ('HVDC', 'AIS'): 1.4,  # HVDC with AIS - high complexity
            ('HVDC', 'GIS'): 1.6,  # HVDC with GIS - highest complexity
            ('HVAC', 'AIS'): 1.0,  # HVAC with AIS - baseline
            ('HVAC', 'GIS'): 1.2   # HVAC with GIS - moderate complexity
        }

        for i in range(1, self.n_projects + 1):
            project_id = "02d"

            # Random project details
            state = random.choice(INDIAN_STATES)
            tower_type = random.choice(TOWER_TYPES)
            substation_type = random.choice(SUBSTATION_TYPES)

            # Get multipliers for this project configuration
            location_multiplier = location_budget_multipliers.get(state, 1.0)
            complexity_key = (tower_type, substation_type)
            complexity_multiplier = complexity_multipliers.get(complexity_key, 1.0)

            # Budget based on project complexity and location
            base_budget_multiplier = random.uniform(50, 200)  # 50-200 crores base
            budget = base_budget_multiplier * 10000000  # Convert to rupees

            # Apply location and complexity multipliers
            budget = budget * location_multiplier * complexity_multiplier

            # Project dates - make some projects current/future
            if i <= 15:  # First 15 projects are current/ongoing
                start_date = datetime.now() - timedelta(days=random.randint(180, 365))  # Started 6-12 months ago
                duration_months = random.randint(24, 48)  # 2-4 years duration
            else:  # Rest are future projects
                start_date = datetime.now() + timedelta(days=random.randint(30, 180))  # Starting in 1-6 months
                duration_months = random.randint(12, 36)

            end_date = start_date + timedelta(days=duration_months * 30)

            projects.append({
                'project_id': project_id,
                'name': f'Power Transmission Project {project_id} - {state}',
                'budget': round(budget, 2),
                'start_date': start_date.date(),
                'end_date': end_date.date(),
                'location': state,
                'tower_type': tower_type,
                'substation_type': substation_type,
                'complexity_multiplier': complexity_multiplier,
                'location_budget_multiplier': location_multiplier
            })

        return pd.DataFrame(projects)

    def generate_materials(self):
        """Generate materials data"""
        materials = []

        selected_materials = random.sample(MATERIALS, self.m_materials)

        for i, material in enumerate(selected_materials, 1):
            material_id = f"{i:03d}"

            materials.append({
                'material_id': material_id,
                'name': material['name'],
                'unit': material['unit'],
                'lead_time_days': random.choice(LEAD_TIME_DAYS),
                'tax_rate': material['tax_rate']
            })

        return pd.DataFrame(materials)

    def generate_seasonal_demand(self, date):
        """Generate seasonal demand pattern"""
        month = date.month

        # Base demand with seasonal variation
        base_demand = 1.0

        # Monsoon season (June-September) - lower construction activity
        if month in [6, 7, 8, 9]:
            seasonal_factor = 0.6
        # Winter (December-February) - moderate activity
        elif month in [12, 1, 2]:
            seasonal_factor = 0.8
        # Summer (March-May) - high activity but heat affects
        elif month in [3, 4, 5]:
            seasonal_factor = 0.9
        # Post-monsoon (October-November) - peak activity
        else:
            seasonal_factor = 1.2

        # Add random spikes (10% chance of 2x demand)
        spike_factor = 2.0 if random.random() < 0.1 else 1.0

        return base_demand * seasonal_factor * spike_factor

    def generate_historical_demand(self, projects_df, materials_df):
        """Generate 36 months of historical demand data"""
        demand_data = []

        # Generate monthly dates for 36 months
        dates = pd.date_range(start=self.start_date, periods=36, freq='M')

        for _, project in projects_df.iterrows():
            for _, material in materials_df.iterrows():
                # Skip some material-project combinations randomly
                if random.random() < 0.3:  # 30% chance of no demand
                    continue

                for date in dates:
                    # Generate demand with seasonal patterns
                    seasonal_factor = self.generate_seasonal_demand(date)

                    # Base demand based on project budget and material type
                    budget_factor = project['budget'] / 100000000  # Normalize budget
                    base_demand = budget_factor * random.uniform(0.5, 2.0)

                    # Apply project complexity multiplier (more complex projects need more materials)
                    complexity_factor = project.get('complexity_multiplier', 1.0)
                    base_demand *= complexity_factor

                    # Material-specific demand variation
                    if material['unit'] == 'km':
                        demand = base_demand * random.uniform(5, 20)  # km of cable
                    elif material['unit'] == 'MT':
                        demand = base_demand * random.uniform(10, 50)  # tons of steel
                    elif material['unit'] == 'units':
                        demand = base_demand * random.uniform(1, 5)  # number of units
                    else:  # pieces
                        demand = base_demand * random.uniform(50, 200)

                    demand *= seasonal_factor

                    # Add some noise
                    demand *= random.uniform(0.8, 1.2)

                    # Ensure minimum demand
                    demand = max(demand, 0.1)

                    demand_data.append({
                        'project_id': project['project_id'],
                        'material_id': material['material_id'],
                        'date': date.date(),
                        'quantity': round(demand, 2)
                    })

        return pd.DataFrame(demand_data)

    def generate_inventory(self, materials_df, current_date=None):
        """Generate current inventory levels"""
        if current_date is None:
            current_date = datetime.now().date()

        inventory_data = []

        for _, material in materials_df.iterrows():
            # Inventory levels based on material type and lead time
            if material['unit'] == 'km':
                base_stock = random.uniform(100, 500)
            elif material['unit'] == 'MT':
                base_stock = random.uniform(50, 200)
            elif material['unit'] == 'units':
                base_stock = random.uniform(5, 20)
            else:  # pieces
                base_stock = random.uniform(1000, 5000)

            # Adjust for lead time (longer lead time = higher safety stock)
            lead_time_factor = material['lead_time_days'] / 30  # months
            safety_stock = base_stock * (1 + lead_time_factor * 0.5)

            inventory_data.append({
                'material_id': material['material_id'],
                'date': current_date,
                'stock_level': round(safety_stock, 2)
            })

        return pd.DataFrame(inventory_data)

    def generate_purchase_orders(self, demand_df, materials_df):
        """Generate purchase orders based on demand patterns"""
        po_data = []

        # Convert date column to datetime for grouping
        demand_df['date'] = pd.to_datetime(demand_df['date'])

        # Group demand by material and month
        monthly_demand = demand_df.groupby(['material_id', pd.Grouper(key='date', freq='M')])['quantity'].sum().reset_index()

        for _, row in monthly_demand.iterrows():
            material_id = row['material_id']
            demand_date = row['date']
            quantity = row['quantity']

            # Generate POs for 20% of demand cases
            if random.random() < 0.2:
                # Order quantity (80-120% of demand)
                order_qty = quantity * random.uniform(0.8, 1.2)

                # Order date (1-3 months before demand)
                lead_time = materials_df[materials_df['material_id'] == material_id]['lead_time_days'].iloc[0]
                order_date = demand_date - timedelta(days=random.randint(lead_time, lead_time + 60))

                # Expected delivery (order_date + lead_time ± 10 days)
                expected_delivery = order_date + timedelta(days=int(lead_time) + random.randint(-10, 10))

                po_data.append({
                    'material_id': material_id,
                    'order_date': order_date.date(),
                    'qty': round(order_qty, 2),
                    'expected_delivery': expected_delivery.date()
                })

        return pd.DataFrame(po_data)

    def generate_pipeline(self, demand_df, projects_df, materials_df):
        """Generate pipeline data (planned future requirements)"""
        pipeline_data = []

        # Look at future demand (next 6-12 months)
        future_start = datetime.now() + timedelta(days=30)
        future_end = datetime.now() + timedelta(days=365)

        future_dates = pd.date_range(start=future_start, end=future_end, freq='M')

        for _, project in projects_df.iterrows():
            # Only include active projects
            if project['end_date'] > datetime.now().date():
                for _, material in materials_df.iterrows():
                    for date in future_dates:
                        # Generate planned quantities (similar to historical but with uncertainty)
                        base_qty = random.uniform(10, 100)

                        # Project phase affects demand
                        project_progress = (datetime.now().date() - project['start_date']).days / (project['end_date'] - project['start_date']).days
                        phase_factor = 1.0 if project_progress < 0.8 else 0.3  # Less demand in final phase

                        planned_qty = base_qty * phase_factor * random.uniform(0.8, 1.2)

                        pipeline_data.append({
                            'project_id': project['project_id'],
                            'material_id': material['material_id'],
                            'planned_qty': round(planned_qty, 2),
                            'delivery_date': date.date()
                        })

        return pd.DataFrame(pipeline_data)

    def generate_all_data(self):
        """Generate all synthetic datasets"""
        print("Generating synthetic POWERGRID data...")

        # Generate base data
        projects_df = self.generate_projects()
        materials_df = self.generate_materials()

        print(f"Generated {len(projects_df)} projects and {len(materials_df)} materials")

        # Generate time-series data
        historical_df = self.generate_historical_demand(projects_df, materials_df)
        inventory_df = self.generate_inventory(materials_df)
        po_df = self.generate_purchase_orders(historical_df, materials_df)
        pipeline_df = self.generate_pipeline(historical_df, projects_df, materials_df)

        print(f"Generated {len(historical_df)} historical demand records")
        print(f"Generated {len(inventory_df)} inventory records")
        print(f"Generated {len(po_df)} purchase order records")
        print(f"Generated {len(pipeline_df)} pipeline records")

        # Save to CSV files
        self.save_to_csv(projects_df, 'projects.csv')
        self.save_to_csv(materials_df, 'materials.csv')
        self.save_to_csv(historical_df, 'historical_demand.csv')
        self.save_to_csv(inventory_df, 'inventory.csv')
        self.save_to_csv(po_df, 'purchase_orders.csv')
        self.save_to_csv(pipeline_df, 'pipeline.csv')

        print(f"\nAll data saved to {self.output_dir}/")

        return {
            'projects': projects_df,
            'materials': materials_df,
            'historical_demand': historical_df,
            'inventory': inventory_df,
            'purchase_orders': po_df,
            'pipeline': pipeline_df
        }

    def save_to_csv(self, df, filename):
        """Save DataFrame to CSV"""
        filepath = self.output_dir / filename
        df.to_csv(filepath, index=False)
        print(f"Saved {len(df)} records to {filepath}")

def main():
    """Main function to generate synthetic data"""
    # Create generator with default parameters
    generator = PowerGridDataGenerator(n_projects=50, m_materials=10)

    # Generate all data
    data = generator.generate_all_data()

    # Print summary statistics
    print("\n" + "="*50)
    print("SYNTHETIC DATA GENERATION COMPLETE")
    print("="*50)
    print(f"Projects: {len(data['projects'])}")
    print(f"Materials: {len(data['materials'])}")
    print(f"Historical Demand Records: {len(data['historical_demand'])}")
    print(f"Date Range: {data['historical_demand']['date'].min()} to {data['historical_demand']['date'].max()}")
    print(f"Total Demand Value: ₹{data['historical_demand']['quantity'].sum():,.0f}")
    print(f"Inventory Records: {len(data['inventory'])}")
    print(f"Purchase Orders: {len(data['purchase_orders'])}")
    print(f"Pipeline Records: {len(data['pipeline'])}")
    print(f"\nFiles saved in: {generator.output_dir}/")

if __name__ == "__main__":
    main()