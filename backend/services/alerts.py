import pandas as pd
import numpy as np
from datetime import datetime, date
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
from dataclasses import dataclass
from enum import Enum

class AlertType(Enum):
    SHORTAGE = "shortage"
    OVERSTOCK = "overstock"
    LATE_DELIVERY = "late_delivery"

class AlertSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class InventoryAlert:
    alert_id: str
    alert_type: AlertType
    severity: AlertSeverity
    material_id: str
    project_id: Optional[str]
    title: str
    description: str
    current_value: float
    threshold_value: float
    recommended_action: str
    created_at: datetime
    metadata: Dict[str, Any]

class AlertsService:
    def __init__(self):
        self.procurement_data = None
        self.forecast_data = None
        self.inventory_data = None
        self._load_data()

    def _load_data(self):
        """Load procurement plan, forecast, and inventory data"""
        try:
            # Load procurement plan
            procurement_path = Path("../ml/outputs/procurement_plan.csv")
            if procurement_path.exists():
                self.procurement_data = pd.read_csv(procurement_path)
                print(f"✅ Loaded procurement data: {len(self.procurement_data)} materials")
            else:
                print("⚠️ Procurement plan not found")

            # Load forecast data
            forecast_path = Path("../ml/outputs/final_forecast.csv")
            if forecast_path.exists():
                self.forecast_data = pd.read_csv(forecast_path)
                self.forecast_data['date'] = pd.to_datetime(self.forecast_data['date'])
                print(f"✅ Loaded forecast data: {len(self.forecast_data)} records")
            else:
                print("⚠️ Forecast data not found")

            # Load inventory data
            inventory_path = Path("../data/inventory.csv")
            if inventory_path.exists():
                self.inventory_data = pd.read_csv(inventory_path)
                print(f"✅ Loaded inventory data: {len(self.inventory_data)} materials")
            else:
                print("⚠️ Inventory data not found")

        except Exception as e:
            print(f"❌ Error loading data: {e}")

    def _calculate_shortage_alerts(self) -> List[InventoryAlert]:
        """Detect shortage alerts: forecast_p50 > stock + incoming_pos"""
        alerts = []

        if self.forecast_data is None or self.inventory_data is None:
            return alerts

        # Get latest forecast for each material (most recent date)
        latest_forecasts = self.forecast_data.sort_values('date').groupby('material_id').last().reset_index()

        for _, forecast in latest_forecasts.iterrows():
            material_id = forecast['material_id']
            forecast_p50 = forecast['p50']

            # Get current inventory
            inventory_row = self.inventory_data[self.inventory_data['material_id'] == material_id]
            if len(inventory_row) == 0:
                continue

            current_stock = inventory_row['current_stock'].iloc[0]

            # Assume no incoming POs for now (can be extended)
            incoming_pos = 0  # TODO: Add incoming purchase orders data

            available_stock = current_stock + incoming_pos

            if forecast_p50 > available_stock:
                shortage_amount = forecast_p50 - available_stock
                shortage_percentage = (shortage_amount / forecast_p50) * 100

                # Determine severity
                if shortage_percentage > 50:
                    severity = AlertSeverity.CRITICAL
                elif shortage_percentage > 25:
                    severity = AlertSeverity.HIGH
                elif shortage_percentage > 10:
                    severity = AlertSeverity.MEDIUM
                else:
                    severity = AlertSeverity.LOW

                alert = InventoryAlert(
                    alert_id=f"shortage_{material_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    alert_type=AlertType.SHORTAGE,
                    severity=severity,
                    material_id=material_id,
                    project_id=None,  # TODO: Add project mapping
                    title=f"Stock Shortage Alert - Material {material_id}",
                    description=f"Forecasted demand ({forecast_p50:.1f}) exceeds available stock ({available_stock:.1f}). "
                               f"Shortage: {shortage_amount:.1f} units ({shortage_percentage:.1f}%).",
                    current_value=available_stock,
                    threshold_value=forecast_p50,
                    recommended_action=f"Place emergency order for {shortage_amount:.0f} units immediately.",
                    created_at=datetime.now(),
                    metadata={
                        'forecast_p50': forecast_p50,
                        'current_stock': current_stock,
                        'incoming_pos': incoming_pos,
                        'shortage_amount': shortage_amount,
                        'shortage_percentage': shortage_percentage,
                        'forecast_date': forecast['date'].isoformat()
                    }
                )
                alerts.append(alert)

        return alerts

    def _calculate_overstock_alerts(self) -> List[InventoryAlert]:
        """Detect overstock alerts: stock > forecast_p50 * 1.5"""
        alerts = []

        if self.forecast_data is None or self.inventory_data is None:
            return alerts

        # Get latest forecast for each material
        latest_forecasts = self.forecast_data.sort_values('date').groupby('material_id').last().reset_index()

        for _, forecast in latest_forecasts.iterrows():
            material_id = forecast['material_id']
            forecast_p50 = forecast['p50']

            # Get current inventory
            inventory_row = self.inventory_data[self.inventory_data['material_id'] == material_id]
            if len(inventory_row) == 0:
                continue

            current_stock = inventory_row['current_stock'].iloc[0]
            overstock_threshold = forecast_p50 * 1.5

            if current_stock > overstock_threshold:
                excess_amount = current_stock - forecast_p50
                excess_percentage = ((current_stock - forecast_p50) / forecast_p50) * 100

                # Determine severity
                if excess_percentage > 100:  # More than double the forecast
                    severity = AlertSeverity.HIGH
                elif excess_percentage > 50:  # 50% over forecast
                    severity = AlertSeverity.MEDIUM
                else:
                    severity = AlertSeverity.LOW

                alert = InventoryAlert(
                    alert_id=f"overstock_{material_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    alert_type=AlertType.OVERSTOCK,
                    severity=severity,
                    material_id=material_id,
                    project_id=None,
                    title=f"Overstock Alert - Material {material_id}",
                    description=f"Current stock ({current_stock:.1f}) exceeds 150% of forecasted demand ({forecast_p50:.1f}). "
                               f"Excess: {excess_amount:.1f} units ({excess_percentage:.1f}%).",
                    current_value=current_stock,
                    threshold_value=overstock_threshold,
                    recommended_action=f"Consider reducing future orders or implementing stock reduction strategies.",
                    created_at=datetime.now(),
                    metadata={
                        'forecast_p50': forecast_p50,
                        'current_stock': current_stock,
                        'overstock_threshold': overstock_threshold,
                        'excess_amount': excess_amount,
                        'excess_percentage': excess_percentage,
                        'forecast_date': forecast['date'].isoformat()
                    }
                )
                alerts.append(alert)

        return alerts

    def _calculate_late_delivery_alerts(self) -> List[InventoryAlert]:
        """Detect late delivery alerts: recommended_order_date < today"""
        alerts = []

        if self.procurement_data is None:
            return alerts

        today = date.today()

        for _, procurement in self.procurement_data.iterrows():
            material_id = procurement['material_id']
            recommended_order_date_str = procurement['recommended_order_date']
            recommended_order_qty = procurement['recommended_order_qty']

            # Skip if no order is recommended
            if recommended_order_qty <= 0:
                continue

            try:
                recommended_order_date = datetime.fromisoformat(recommended_order_date_str).date()
            except:
                continue

            if recommended_order_date < today:
                days_late = (today - recommended_order_date).days

                # Determine severity based on how late
                if days_late > 30:
                    severity = AlertSeverity.CRITICAL
                elif days_late > 14:
                    severity = AlertSeverity.HIGH
                elif days_late > 7:
                    severity = AlertSeverity.MEDIUM
                else:
                    severity = AlertSeverity.LOW

                alert = InventoryAlert(
                    alert_id=f"late_delivery_{material_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    alert_type=AlertType.LATE_DELIVERY,
                    severity=severity,
                    material_id=material_id,
                    project_id=None,
                    title=f"Late Delivery Alert - Material {material_id}",
                    description=f"Recommended order date was {recommended_order_date} ({days_late} days ago). "
                               f"Order quantity: {recommended_order_qty:.1f} units.",
                    current_value=days_late,
                    threshold_value=0,
                    recommended_action=f"Place order immediately for {recommended_order_qty:.0f} units. "
                                     f"Contact supplier to expedite delivery.",
                    created_at=datetime.now(),
                    metadata={
                        'recommended_order_date': recommended_order_date_str,
                        'recommended_order_qty': recommended_order_qty,
                        'days_late': days_late,
                        'order_value': procurement.get('order_value', 0),
                        'stockout_risk': procurement.get('stockout_risk', 0)
                    }
                )
                alerts.append(alert)

        return alerts

    def get_alerts(self, project_id: Optional[str] = None, material_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all inventory alerts with optional filtering

        Args:
            project_id: Filter by project ID (optional)
            material_id: Filter by material ID (optional)

        Returns:
            List of alert dictionaries
        """
        # Refresh data
        self._load_data()

        # Generate all alerts
        shortage_alerts = self._calculate_shortage_alerts()
        overstock_alerts = self._calculate_overstock_alerts()
        late_delivery_alerts = self._calculate_late_delivery_alerts()

        all_alerts = shortage_alerts + overstock_alerts + late_delivery_alerts

        # Convert to dictionaries
        alert_dicts = []
        for alert in all_alerts:
            # Apply filters
            if project_id and alert.project_id != project_id:
                continue
            if material_id and alert.material_id != material_id:
                continue

            alert_dict = {
                'alert_id': alert.alert_id,
                'alert_type': alert.alert_type.value,
                'severity': alert.severity.value,
                'material_id': alert.material_id,
                'project_id': alert.project_id,
                'title': alert.title,
                'description': alert.description,
                'current_value': alert.current_value,
                'threshold_value': alert.threshold_value,
                'recommended_action': alert.recommended_action,
                'created_at': alert.created_at.isoformat(),
                'metadata': alert.metadata
            }
            alert_dicts.append(alert_dict)

        # Sort by severity (critical first) and creation time
        severity_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        alert_dicts.sort(key=lambda x: (severity_order.get(x['severity'], 4), x['created_at']), reverse=True)

        return alert_dicts

    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary statistics of alerts"""
        all_alerts = self.get_alerts()

        summary = {
            'total_alerts': len(all_alerts),
            'by_type': {},
            'by_severity': {},
            'by_material': {},
            'generated_at': datetime.now().isoformat()
        }

        for alert in all_alerts:
            # Count by type
            alert_type = alert['alert_type']
            summary['by_type'][alert_type] = summary['by_type'].get(alert_type, 0) + 1

            # Count by severity
            severity = alert['severity']
            summary['by_severity'][severity] = summary['by_severity'].get(severity, 0) + 1

            # Count by material
            material = alert['material_id']
            summary['by_material'][material] = summary['by_material'].get(material, 0) + 1

        return summary

# Global service instance
alerts_service = AlertsService()