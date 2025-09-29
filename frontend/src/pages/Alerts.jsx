import { useState, useEffect } from "react";
import { apiService } from "../services/api";

const Alerts = () => {
  const [alerts, setAlerts] = useState([]);
  const [summary, setSummary] = useState({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [filters, setFilters] = useState({
    project_id: "",
    material_id: "",
    severity: "",
  });

  useEffect(() => {
    loadAlerts();
  }, [filters]);

  const loadAlerts = async () => {
    try {
      setLoading(true);
      setError(null);

      // Load alerts with filters
      const params = {};
      if (filters.project_id) params.project_id = filters.project_id;
      if (filters.material_id) params.material_id = filters.material_id;

      const [alertsResponse, summaryResponse] = await Promise.all([
        apiService.getAlerts(params),
        apiService.getAlertsSummary(),
      ]);

      // Filter by severity if specified
      let filteredAlerts = alertsResponse.data;
      if (filters.severity) {
        filteredAlerts = filteredAlerts.filter(
          (alert) => alert.severity === filters.severity
        );
      }

      setAlerts(filteredAlerts);
      setSummary(summaryResponse.data);
    } catch (err) {
      console.error("Error loading alerts:", err);
      setError(
        "Failed to load alerts. Please check if the backend is running."
      );
    } finally {
      setLoading(false);
    }
  };

  const getSeverityColor = (severity) => {
    switch (severity) {
      case "critical":
        return "bg-red-100 text-red-800 border-red-200";
      case "high":
        return "bg-orange-100 text-orange-800 border-orange-200";
      case "medium":
        return "bg-yellow-100 text-yellow-800 border-yellow-200";
      case "low":
        return "bg-green-100 text-green-800 border-green-200";
      default:
        return "bg-gray-100 text-gray-800 border-gray-200";
    }
  };

  const getAlertTypeIcon = (alertType) => {
    switch (alertType) {
      case "shortage":
        return "📉";
      case "overstock":
        return "📈";
      case "late_delivery":
        return "⏰";
      default:
        return "⚠️";
    }
  };

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleDateString("en-US", {
      year: "numeric",
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading alerts...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="bg-red-50 border border-red-200 rounded-lg p-6 max-w-md">
            <h2 className="text-red-800 font-semibold mb-2">
              Error Loading Alerts
            </h2>
            <p className="text-red-600">{error}</p>
            <button
              onClick={loadAlerts}
              className="mt-4 bg-red-600 text-white px-4 py-2 rounded-md hover:bg-red-700"
            >
              Retry
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900">Inventory Alerts</h1>
          <p className="mt-2 text-gray-600">
            Monitor inventory risks and take proactive actions.
          </p>
        </div>

        {/* Summary Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-6">
          <div className="bg-white rounded-lg shadow-md p-6">
            <h3 className="text-lg font-semibold text-gray-900">
              Total Alerts
            </h3>
            <p className="text-3xl font-bold text-blue-600">
              {summary.total_alerts || 0}
            </p>
          </div>
          <div className="bg-white rounded-lg shadow-md p-6">
            <h3 className="text-lg font-semibold text-gray-900">Critical</h3>
            <p className="text-3xl font-bold text-red-600">
              {summary.by_severity?.critical || 0}
            </p>
          </div>
          <div className="bg-white rounded-lg shadow-md p-6">
            <h3 className="text-lg font-semibold text-gray-900">
              High Priority
            </h3>
            <p className="text-3xl font-bold text-orange-600">
              {summary.by_severity?.high || 0}
            </p>
          </div>
          <div className="bg-white rounded-lg shadow-md p-6">
            <h3 className="text-lg font-semibold text-gray-900">
              Materials at Risk
            </h3>
            <p className="text-3xl font-bold text-purple-600">
              {Object.keys(summary.by_material || {}).length}
            </p>
          </div>
        </div>

        {/* Filters */}
        <div className="bg-white rounded-lg shadow-md p-6 mb-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Filters</h3>
          <div className="flex flex-wrap gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Project ID
              </label>
              <input
                type="text"
                value={filters.project_id}
                onChange={(e) =>
                  setFilters((prev) => ({
                    ...prev,
                    project_id: e.target.value,
                  }))
                }
                placeholder="Enter project ID"
                className="border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Material ID
              </label>
              <input
                type="text"
                value={filters.material_id}
                onChange={(e) =>
                  setFilters((prev) => ({
                    ...prev,
                    material_id: e.target.value,
                  }))
                }
                placeholder="Enter material ID"
                className="border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Severity
              </label>
              <select
                value={filters.severity}
                onChange={(e) =>
                  setFilters((prev) => ({ ...prev, severity: e.target.value }))
                }
                className="border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="">All Severities</option>
                <option value="critical">Critical</option>
                <option value="high">High</option>
                <option value="medium">Medium</option>
                <option value="low">Low</option>
              </select>
            </div>

            <div className="flex items-end">
              <button
                onClick={() =>
                  setFilters({ project_id: "", material_id: "", severity: "" })
                }
                className="bg-gray-500 text-white px-4 py-2 rounded-md hover:bg-gray-600"
              >
                Clear Filters
              </button>
            </div>
          </div>
        </div>

        {/* Alerts Table */}
        <div className="bg-white rounded-lg shadow-md overflow-hidden">
          <div className="px-6 py-4 border-b border-gray-200">
            <h3 className="text-lg font-semibold text-gray-900">
              Active Alerts ({alerts.length})
            </h3>
          </div>

          {alerts.length === 0 ? (
            <div className="p-6 text-center text-gray-500">
              <p>No alerts found matching the current filters.</p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Type
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Severity
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Material
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Title
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Description
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Action Required
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Created
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {alerts.map((alert) => (
                    <tr key={alert.alert_id} className="hover:bg-gray-50">
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="flex items-center">
                          <span className="text-lg mr-2">
                            {getAlertTypeIcon(alert.alert_type)}
                          </span>
                          <span className="text-sm font-medium text-gray-900 capitalize">
                            {alert.alert_type.replace("_", " ")}
                          </span>
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span
                          className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full border ${getSeverityColor(
                            alert.severity
                          )}`}
                        >
                          {alert.severity.toUpperCase()}
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                        {alert.material_id}
                      </td>
                      <td className="px-6 py-4 text-sm text-gray-900 max-w-xs">
                        <div className="font-medium">{alert.title}</div>
                      </td>
                      <td className="px-6 py-4 text-sm text-gray-500 max-w-md">
                        <div className="line-clamp-2">{alert.description}</div>
                      </td>
                      <td className="px-6 py-4 text-sm text-gray-900 max-w-sm">
                        <div
                          className={`p-2 rounded-md ${
                            alert.severity === "critical"
                              ? "bg-red-50 text-red-800"
                              : alert.severity === "high"
                              ? "bg-orange-50 text-orange-800"
                              : "bg-blue-50 text-blue-800"
                          }`}
                        >
                          {alert.recommended_action}
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {formatDate(alert.created_at)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* Alert Type Information */}
        <div className="mt-8 bg-blue-50 border border-blue-200 rounded-lg p-6">
          <h3 className="text-lg font-medium text-blue-900 mb-4">
            Alert Types
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-white p-4 rounded-md">
              <div className="flex items-center mb-2">
                <span className="text-lg mr-2">📉</span>
                <h4 className="font-medium text-gray-900">Shortage Alerts</h4>
              </div>
              <p className="text-sm text-gray-600">
                Triggered when forecasted demand exceeds available stock.
              </p>
            </div>
            <div className="bg-white p-4 rounded-md">
              <div className="flex items-center mb-2">
                <span className="text-lg mr-2">📈</span>
                <h4 className="font-medium text-gray-900">Overstock Alerts</h4>
              </div>
              <p className="text-sm text-gray-600">
                Triggered when stock exceeds 150% of forecasted demand.
              </p>
            </div>
            <div className="bg-white p-4 rounded-md">
              <div className="flex items-center mb-2">
                <span className="text-lg mr-2">⏰</span>
                <h4 className="font-medium text-gray-900">
                  Late Delivery Alerts
                </h4>
              </div>
              <p className="text-sm text-gray-600">
                Triggered when recommended order dates are in the past.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Alerts;
