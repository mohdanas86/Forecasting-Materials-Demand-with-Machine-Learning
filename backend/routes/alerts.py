from fastapi import APIRouter, Query, HTTPException, Depends
from typing import List, Dict, Any, Optional
from services.alerts import alerts_service
from auth import get_current_user_optional

router = APIRouter()

@router.get("/alerts", response_model=List[Dict[str, Any]])
async def get_alerts(
    project_id: Optional[str] = Query(None, description="Filter by project ID"),
    material_id: Optional[str] = Query(None, description="Filter by material ID"),
    user_id: Optional[str] = Depends(get_current_user_optional)
):
    """
    Get inventory alerts with optional filtering

    - **project_id**: Filter alerts by project ID
    - **material_id**: Filter alerts by material ID

    Returns a list of alerts with the following information:
    - alert_id: Unique identifier for the alert
    - alert_type: Type of alert (shortage, overstock, late_delivery)
    - severity: Alert severity (low, medium, high, critical)
    - material_id: Material identifier
    - title: Alert title
    - description: Detailed alert description
    - recommended_action: Suggested action to resolve the alert
    - metadata: Additional alert-specific data
    """
    try:
        alerts = alerts_service.get_alerts(project_id=project_id, material_id=material_id)
        return alerts
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving alerts: {str(e)}")

@router.get("/alerts/summary")
async def get_alerts_summary(user_id: Optional[str] = Depends(get_current_user_optional)):
    """
    Get summary statistics of inventory alerts

    Returns summary information including:
    - total_alerts: Total number of active alerts
    - by_type: Alert count by type (shortage, overstock, late_delivery)
    - by_severity: Alert count by severity level
    - by_material: Alert count by material
    """
    try:
        summary = alerts_service.get_alert_summary()
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving alert summary: {str(e)}")

@router.get("/alerts/types")
async def get_alert_types(user_id: Optional[str] = Depends(get_current_user_optional)):
    """
    Get information about available alert types

    Returns descriptions of each alert type and their detection criteria.
    """
    alert_types_info = {
        "shortage": {
            "description": "Stock shortage alerts",
            "criteria": "Triggered when forecasted demand (p50) exceeds available stock + incoming purchase orders",
            "severity_levels": ["low", "medium", "high", "critical"],
            "recommended_action": "Place emergency order or expedite existing orders"
        },
        "overstock": {
            "description": "Overstock alerts",
            "criteria": "Triggered when current stock exceeds 150% of forecasted demand (p50)",
            "severity_levels": ["low", "medium", "high"],
            "recommended_action": "Consider reducing future orders or implementing stock reduction strategies"
        },
        "late_delivery": {
            "description": "Late delivery alerts",
            "criteria": "Triggered when recommended order date is in the past",
            "severity_levels": ["low", "medium", "high", "critical"],
            "recommended_action": "Place order immediately and contact supplier to expedite delivery"
        }
    }

    return alert_types_info