from fastapi import APIRouter, Query, HTTPException, Depends
from typing import List, Dict, Any, Optional
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from auth import get_current_user_optional

router = APIRouter()

router = APIRouter()

def load_forecast_data():
    """Load forecast and procurement data"""
    try:
        # Load final forecast
        forecast_path = Path("../ml/outputs/final_forecast.csv")
        if not forecast_path.exists():
            raise HTTPException(status_code=404, detail="Forecast data not found")

        forecast_df = pd.read_csv(forecast_path)
        forecast_df['date'] = pd.to_datetime(forecast_df['date'])

        # Load procurement plan
        procurement_path = Path("../ml/outputs/procurement_plan.csv")
        procurement_df = None
        if procurement_path.exists():
            procurement_df = pd.read_csv(procurement_path)

        return forecast_df, procurement_df

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading data: {str(e)}")

@router.get("/forecast")
async def get_forecast(
    project_id: Optional[str] = Query(None, description="Filter by project ID"),
    material_id: Optional[str] = Query(None, description="Filter by material ID"),
    location: Optional[str] = Query(None, description="Filter by location/state"),
    tower_type: Optional[str] = Query(None, description="Filter by tower type"),
    substation_type: Optional[str] = Query(None, description="Filter by substation type"),
    budget_min: Optional[float] = Query(None, description="Minimum budget/cost filter"),
    budget_max: Optional[float] = Query(None, description="Maximum budget/cost filter"),
    period: Optional[str] = Query("all", description="Time period: 'week', 'month', 'quarter', 'all'"),
    user_id: Optional[str] = Depends(get_current_user_optional)
):
    """
    Get forecast data with procurement metrics

    - **project_id**: Filter by project ID
    - **material_id**: Filter by material ID
    - **location**: Filter by location/state
    - **tower_type**: Filter by tower type (Lattice Tower, Tubular Tower, etc.)
    - **substation_type**: Filter by substation type (AIS Substation, GIS Substation, etc.)
    - **budget_min**: Minimum budget/cost filter
    - **budget_max**: Maximum budget/cost filter
    - **period**: Time period filter ('week', 'month', 'quarter', 'all')

    Returns forecast data with p10, p50, p90 values, safety stock, reorder point, and recommendations.
    """
    try:
        forecast_df, procurement_df = load_forecast_data()

        # Apply filters
        if material_id:
            forecast_df = forecast_df[forecast_df['material_id'] == material_id]

        # Note: Location, tower_type, and substation_type filters are not available
        # without features data. These parameters are kept for future implementation.

        # Apply budget filters using procurement data
        if procurement_df is not None and (budget_min is not None or budget_max is not None):
            # Filter materials by total_cost from procurement plan
            filtered_materials = procurement_df['material_id'].copy()

            if budget_min is not None:
                filtered_materials = procurement_df[procurement_df['total_cost'] >= budget_min]['material_id']

            if budget_max is not None:
                filtered_materials = procurement_df[procurement_df['total_cost'] <= budget_max]['material_id']

            if budget_min is not None and budget_max is not None:
                filtered_materials = procurement_df[
                    (procurement_df['total_cost'] >= budget_min) &
                    (procurement_df['total_cost'] <= budget_max)
                ]['material_id']

            # Apply material filter to forecast data
            forecast_df = forecast_df[forecast_df['material_id'].isin(filtered_materials)]

        # Apply time period filter
        if period != "all":
            today = datetime.now().date()
            if period == "week":
                cutoff_date = today + timedelta(days=7)
            elif period == "month":
                cutoff_date = today + timedelta(days=30)
            elif period == "quarter":
                cutoff_date = today + timedelta(days=90)
            else:
                cutoff_date = None

            if cutoff_date:
                forecast_df = forecast_df[forecast_df['date'].dt.date <= cutoff_date]

        # Prepare response data
        forecast_results = []

        # Group by material for aggregation
        for material_id_val in forecast_df['material_id'].unique():
            material_forecast = forecast_df[forecast_df['material_id'] == material_id_val]

            # Get latest forecast values
            latest_forecast = material_forecast.sort_values('date').iloc[-1] if len(material_forecast) > 0 else None

            if latest_forecast is None:
                continue

            # Get procurement data for this material
            procurement_data = None
            if procurement_df is not None:
                material_procurement = procurement_df[procurement_df['material_id'] == material_id_val]
                if len(material_procurement) > 0:
                    procurement_data = material_procurement.iloc[0].to_dict()

            # Calculate recommendations based on current stock vs forecast
            recommendations = []

            if procurement_data:
                current_stock = procurement_data.get('current_stock', 0)
                reorder_point = procurement_data.get('reorder_point', 0)
                safety_stock = procurement_data.get('safety_stock', 0)
                recommended_order_qty = procurement_data.get('recommended_order_qty', 0)

                if current_stock < safety_stock:
                    recommendations.append({
                        "type": "critical",
                        "message": f"Stock below safety level. Current: {current_stock:.1f}, Safety: {safety_stock:.1f}"
                    })

                if current_stock < reorder_point:
                    recommendations.append({
                        "type": "urgent",
                        "message": f"Reorder point reached. Consider ordering {recommended_order_qty:.0f} units"
                    })

                if recommended_order_qty > 0:
                    recommendations.append({
                        "type": "action",
                        "message": f"Recommended order: {recommended_order_qty:.0f} units by {procurement_data.get('recommended_order_date', 'ASAP')}"
                    })

            # Build forecast response with requested schema
            forecast_item = {
                "material_id": material_id_val,
                "project_id": project_id,  # TODO: Add project mapping
                "date": latest_forecast['date'].isoformat(),
                "p10": round(latest_forecast['p10'], 2),
                "p50": round(latest_forecast['p50'], 2),
                "p90": round(latest_forecast['p90'], 2),
                "safety_stock": round(procurement_data.get('safety_stock', 0), 2) if procurement_data else 0,
                "reorder_point": round(procurement_data.get('reorder_point', 0), 2) if procurement_data else 0,
                "recommended_qty": round(procurement_data.get('recommended_order_qty', 0), 2) if procurement_data else 0,
                "recommended_order_date": procurement_data.get('recommended_order_date', None) if procurement_data else None
            }

            forecast_results.append(forecast_item)

        # Sort by material_id
        forecast_results.sort(key=lambda x: x['material_id'])

        return {
            "forecasts": forecast_results,
            "metadata": {
                "total_materials": len(forecast_results),
                "period_filter": period,
                "generated_at": datetime.now().isoformat(),
                "data_source": "ensemble_forecasting_model"
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving forecast: {str(e)}")

@router.get("/forecast/summary")
async def get_forecast_summary(user_id: Optional[str] = Depends(get_current_user_optional)):
    """
    Get summary statistics for forecast data
    """
    try:
        forecast_df, procurement_df = load_forecast_data()

        # Calculate summary statistics
        summary = {
            "total_materials": forecast_df['material_id'].nunique(),
            "total_forecast_points": len(forecast_df),
            "date_range": {
                "start": forecast_df['date'].min().isoformat(),
                "end": forecast_df['date'].max().isoformat()
            },
            "average_forecast": {
                "p10": round(forecast_df['p10'].mean(), 2),
                "p50": round(forecast_df['p50'].mean(), 2),
                "p90": round(forecast_df['p90'].mean(), 2)
            },
            "forecast_volatility": round(forecast_df['p90'].std() / forecast_df['p50'].mean(), 3),
            "generated_at": datetime.now().isoformat()
        }

        # Add procurement summary if available
        if procurement_df is not None:
            summary["procurement"] = {
                "total_materials": len(procurement_df),
                "total_procurement_value": round(procurement_df['total_cost'].sum(), 2),
                "average_cost_per_material": round(procurement_df['total_cost'].mean(), 2),
                "total_quantity": round(procurement_df['qty'].sum(), 2)
            }

        return summary

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving forecast summary: {str(e)}")