import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

def load_forecast_data():
    """Load final forecast data"""
    print("📥 Loading forecast data...")

    forecast_path = Path("ml/outputs/final_forecast.csv")
    if not forecast_path.exists():
        raise FileNotFoundError("Final forecast file not found")

    df = pd.read_csv(forecast_path)
    df['date'] = pd.to_datetime(df['date'])

    print(f"✅ Loaded {len(df)} forecast records for {df['material_id'].nunique()} materials")
    return df

def load_inventory_data():
    """Load current inventory data from database or file"""
    print("📦 Loading inventory data...")

    # Try to load from CSV first (for development/testing)
    inventory_path = Path("data/inventory.csv")
    if inventory_path.exists():
        df = pd.read_csv(inventory_path)
        print(f"✅ Loaded inventory from CSV: {len(df)} records")
        return df

    # If no CSV, try to load from database
    try:
        from motor.motor_asyncio import AsyncIOMotorClient
        import asyncio

        async def get_inventory_from_db():
            client = AsyncIOMotorClient("mongodb://localhost:27017")
            db = client.inventory_db
            collection = db.inventory

            cursor = collection.find({})
            inventory_data = []
            async for document in cursor:
                # Convert ObjectId to string if needed
                document['_id'] = str(document['_id'])
                inventory_data.append(document)

            client.close()
            return pd.DataFrame(inventory_data)

        # Run async function
        df = asyncio.run(get_inventory_from_db())
        print(f"✅ Loaded inventory from database: {len(df)} records")
        return df

    except Exception as e:
        print(f"⚠️ Could not load from database: {e}")
        print("Creating sample inventory data for demonstration...")

        # Create sample inventory data based on materials in forecast
        forecast_df = load_forecast_data()
        materials = forecast_df['material_id'].unique()

        sample_inventory = []
        np.random.seed(42)  # For reproducible results

        for material in materials:
            # Generate realistic inventory levels based on forecast data
            material_forecasts = forecast_df[forecast_df['material_id'] == material]
            avg_demand = material_forecasts['actual_quantity'].mean()

            # Current stock: somewhere between 0.5x to 2x average demand
            current_stock = np.random.uniform(0.5 * avg_demand, 2 * avg_demand)

            sample_inventory.append({
                'material_id': material,
                'current_stock': round(current_stock, 2),
                'unit_cost': np.random.uniform(10, 1000),  # Sample cost per unit
                'supplier_id': f"SUP_{np.random.randint(1, 11)}",
                'last_updated': datetime.now().isoformat()
            })

        df = pd.DataFrame(sample_inventory)
        print(f"✅ Created sample inventory: {len(df)} records")

        # Save sample inventory for future use
        df.to_csv(inventory_path, index=False)
        print(f"💾 Saved sample inventory to {inventory_path}")

        return df

def calculate_procurement_metrics(forecast_df, inventory_df, lead_time_days=30, z_score=1.65):
    """
    Calculate procurement metrics for each material

    Args:
        forecast_df: DataFrame with forecast data (material_id, date, p10, p50, p90, actual_quantity)
        inventory_df: DataFrame with inventory data (material_id, current_stock, etc.)
        lead_time_days: Lead time in days for procurement
        z_score: Z-score for safety stock calculation (1.65 = 95% service level)
    """
    print("🧮 Calculating procurement metrics...")

    # Group forecast data by material
    material_groups = forecast_df.groupby('material_id')

    procurement_plan = []

    for material_id, group in material_groups:
        print(f"📊 Processing material: {material_id}")

        # Get actual demand data (use actual_quantity as historical demand)
        demand_data = group['actual_quantity'].dropna()

        if len(demand_data) == 0:
            print(f"⚠️ No demand data for material {material_id}, skipping...")
            continue

        # Calculate demand statistics
        mean_demand = demand_data.mean()
        std_demand = demand_data.std()

        # Handle case where std is 0 or NaN
        if pd.isna(std_demand) or std_demand == 0:
            std_demand = mean_demand * 0.1  # Assume 10% variability if no variance

        # Get current inventory
        inventory_row = inventory_df[inventory_df['material_id'] == material_id]
        if len(inventory_row) == 0:
            print(f"⚠️ No inventory data for material {material_id}, using 0")
            current_stock = 0
            unit_cost = 100  # Default cost
        else:
            current_stock = inventory_row['current_stock'].iloc[0]
            unit_cost = inventory_row.get('unit_cost', 100).iloc[0] if 'unit_cost' in inventory_row.columns else 100

        # Calculate safety stock (z * std_demand)
        safety_stock = z_score * std_demand

        # Calculate reorder point (mean_demand * lead_time_days + safety_stock)
        reorder_point = (mean_demand * lead_time_days) + safety_stock

        # Calculate recommended order quantity (max(0, reorder_point - current_stock))
        recommended_order_qty = max(0, reorder_point - current_stock)

        # Calculate recommended order date (today - lead_time_days)
        today = datetime.now().date()
        recommended_order_date = today - timedelta(days=lead_time_days)

        # Calculate additional metrics
        forecasted_demand_next_period = group['p50'].iloc[-1] if len(group) > 0 else mean_demand
        stockout_risk = 1 - (current_stock / reorder_point) if reorder_point > 0 else 1
        stockout_risk = max(0, min(1, stockout_risk))  # Clamp between 0 and 1

        # Calculate order value
        order_value = recommended_order_qty * unit_cost

        procurement_plan.append({
            'material_id': material_id,
            'current_stock': round(current_stock, 2),
            'mean_demand': round(mean_demand, 2),
            'std_demand': round(std_demand, 2),
            'safety_stock': round(safety_stock, 2),
            'reorder_point': round(reorder_point, 2),
            'recommended_order_qty': round(recommended_order_qty, 2),
            'recommended_order_date': recommended_order_date.isoformat(),
            'lead_time_days': lead_time_days,
            'z_score': z_score,
            'unit_cost': round(unit_cost, 2),
            'order_value': round(order_value, 2),
            'stockout_risk': round(stockout_risk, 3),
            'forecasted_demand_next_period': round(forecasted_demand_next_period, 2),
            'service_level_target': '95%',
            'calculation_date': datetime.now().isoformat()
        })

    procurement_df = pd.DataFrame(procurement_plan)
    print(f"✅ Calculated procurement metrics for {len(procurement_df)} materials")

    return procurement_df

def generate_procurement_summary(procurement_df):
    """Generate summary statistics for the procurement plan"""
    print("📊 Generating procurement summary...")

    summary = {
        'total_materials': len(procurement_df),
        'materials_needing_orders': len(procurement_df[procurement_df['recommended_order_qty'] > 0]),
        'total_order_value': round(procurement_df['order_value'].sum(), 2),
        'avg_stockout_risk': round(procurement_df['stockout_risk'].mean(), 3),
        'high_risk_materials': len(procurement_df[procurement_df['stockout_risk'] > 0.7]),
        'total_recommended_quantity': round(procurement_df['recommended_order_qty'].sum(), 2),
        'avg_lead_time': procurement_df['lead_time_days'].mean(),
        'calculation_timestamp': datetime.now().isoformat()
    }

    return summary

def save_procurement_plan(procurement_df, summary):
    """Save procurement plan and summary to files"""
    print("💾 Saving procurement plan...")

    # Create output directory
    output_dir = Path("ml/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save detailed procurement plan
    plan_path = output_dir / "procurement_plan.csv"
    procurement_df.to_csv(plan_path, index=False)
    print(f"✅ Procurement plan saved to {plan_path}")

    # Save summary
    summary_path = output_dir / "procurement_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Procurement summary saved to {summary_path}")

    # Save human-readable summary report
    report_path = output_dir / "procurement_report.md"
    with open(report_path, 'w') as f:
        f.write("# Procurement Plan Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Summary\n\n")
        f.write(f"- **Total Materials:** {summary['total_materials']}\n")
        f.write(f"- **Materials Needing Orders:** {summary['materials_needing_orders']}\n")
        f.write(f"- **Total Order Value:** ${summary['total_order_value']:,.2f}\n")
        f.write(f"- **Average Stockout Risk:** {summary['avg_stockout_risk']:.1%}\n")
        f.write(f"- **High Risk Materials (>70% stockout risk):** {summary['high_risk_materials']}\n")
        f.write(f"- **Total Recommended Quantity:** {summary['total_recommended_quantity']:,.2f}\n")
        f.write(f"- **Average Lead Time:** {summary['avg_lead_time']:.0f} days\n\n")

        f.write("## High Priority Orders\n\n")
        high_priority = procurement_df[procurement_df['recommended_order_qty'] > 0].sort_values('stockout_risk', ascending=False).head(10)
        if len(high_priority) > 0:
            f.write("| Material | Current Stock | Recommended Order | Stockout Risk | Order Value |\n")
            f.write("|----------|---------------|-------------------|---------------|-------------|\n")
            for _, row in high_priority.iterrows():
                f.write(f"| {row['material_id']} | {row['current_stock']:,.0f} | {row['recommended_order_qty']:,.0f} | {row['stockout_risk']:.1%} | ${row['order_value']:,.2f} |\n")
        else:
            f.write("No materials currently require ordering.\n")

    print(f"✅ Procurement report saved to {report_path}")

def print_procurement_summary(procurement_df, summary):
    """Print procurement summary to console"""
    print("\n" + "="*70)
    print("PROCUREMENT PLAN SUMMARY")
    print("="*70)
    print(f"Total Materials: {summary['total_materials']}")
    print(f"Materials Needing Orders: {summary['materials_needing_orders']}")
    print(f"Total Order Value: ${summary['total_order_value']:,.2f}")
    print(f"Average Stockout Risk: {summary['avg_stockout_risk']:.1%}")
    print(f"High Risk Materials (>70% stockout risk): {summary['high_risk_materials']}")
    print(f"Total Recommended Quantity: {summary['total_recommended_quantity']:,.2f}")
    print(f"Average Lead Time: {summary['avg_lead_time']:.0f} days")
    print("="*70)

    # Show top 5 materials needing orders
    needing_orders = procurement_df[procurement_df['recommended_order_qty'] > 0]
    if len(needing_orders) > 0:
        print("\nTop 5 Materials Needing Orders:")
        print("-" * 50)
        top_5 = needing_orders.sort_values('stockout_risk', ascending=False).head(5)
        for _, row in top_5.iterrows():
            print(f"Material {row['material_id']}: Order {row['recommended_order_qty']:,.0f} units "
                  f"(Stockout risk: {row['stockout_risk']:.1%})")
    else:
        print("\n✅ All materials have sufficient inventory - no orders needed!")

def main():
    """Main function to generate procurement plan"""
    print("🚀 Starting Procurement Planning Pipeline")
    print("=" * 50)

    try:
        # Configuration
        LEAD_TIME_DAYS = 30  # Configurable lead time
        Z_SCORE = 1.65  # 95% service level (1.65 standard deviations)

        print(f"⚙️ Configuration: Lead time = {LEAD_TIME_DAYS} days, Z-score = {Z_SCORE} (95% service level)")

        # Load data
        forecast_df = load_forecast_data()
        inventory_df = load_inventory_data()

        # Calculate procurement metrics
        procurement_df = calculate_procurement_metrics(
            forecast_df, inventory_df,
            lead_time_days=LEAD_TIME_DAYS,
            z_score=Z_SCORE
        )

        # Generate summary
        summary = generate_procurement_summary(procurement_df)

        # Print and save results
        print_procurement_summary(procurement_df, summary)
        save_procurement_plan(procurement_df, summary)

        print("\n" + "=" * 50)
        print("✅ Procurement planning pipeline completed!")
        print("=" * 50)

    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()