import { useState, useEffect } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  BarChart,
  Bar,
  Area,
  AreaChart,
} from "recharts";
import { apiService } from "../services/api";

const ForecastDashboard = () => {
  const [forecastData, setForecastData] = useState([]);
  const [summaryData, setSummaryData] = useState({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedMaterial, setSelectedMaterial] = useState("");
  const [selectedPeriod, setSelectedPeriod] = useState("all");

  useEffect(() => {
    loadForecastData();
  }, [selectedMaterial, selectedPeriod]);

  const loadForecastData = async () => {
    try {
      setLoading(true);
      setError(null);

      // Load forecast data
      const params = {};
      if (selectedMaterial) params.material_id = selectedMaterial;
      if (selectedPeriod !== "all") params.period = selectedPeriod;

      const forecastResponse = await apiService.getForecast(params);
      setForecastData(forecastResponse.data.forecasts);

      // Load summary data
      const summaryResponse = await apiService.getForecastSummary();
      setSummaryData(summaryResponse.data);
    } catch (err) {
      console.error("Error loading forecast data:", err);
      setError(
        "Failed to load forecast data. Please check if the backend is running."
      );
    } finally {
      setLoading(false);
    }
  };

  const prepareChartData = () => {
    if (!forecastData.length) return [];

    // For demo purposes, create sample historical data
    // In a real app, this would come from the API
    const chartData = [];

    forecastData.forEach((forecast, index) => {
      // Create sample historical data points leading up to forecast
      const baseDate = new Date(forecast.forecast_date);
      for (let i = -30; i <= 0; i++) {
        const date = new Date(baseDate);
        date.setDate(date.getDate() + i);

        chartData.push({
          date: date.toISOString().split("T")[0],
          material_id: forecast.material_id,
          historical: Math.max(0, forecast.p50 * (0.8 + Math.random() * 0.4)), // Simulated historical data
          forecast_p10: null,
          forecast_p50: null,
          forecast_p90: null,
          is_forecast: false,
        });
      }

      // Add forecast data point
      chartData.push({
        date: forecast.forecast_date.split("T")[0],
        material_id: forecast.material_id,
        historical: null,
        forecast_p10: forecast.p10,
        forecast_p50: forecast.p50,
        forecast_p90: forecast.p90,
        is_forecast: true,
      });
    });

    return chartData;
  };

  const prepareSummaryChartData = () => {
    if (!forecastData.length) return [];

    return forecastData.map((forecast) => ({
      material: forecast.material_id,
      p10: forecast.p10,
      p50: forecast.p50,
      p90: forecast.p90,
      range: forecast.p90 - forecast.p10,
      safety_stock: forecast.safety_stock || 0,
      reorder_point: forecast.reorder_point || 0,
    }));
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading forecast data...</p>
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
              Error Loading Data
            </h2>
            <p className="text-red-600">{error}</p>
            <button
              onClick={loadForecastData}
              className="mt-4 bg-red-600 text-white px-4 py-2 rounded-md hover:bg-red-700"
            >
              Retry
            </button>
          </div>
        </div>
      </div>
    );
  }

  const chartData = prepareChartData();
  const summaryChartData = prepareSummaryChartData();

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900">
            Forecast Dashboard
          </h1>
          <p className="mt-2 text-gray-600">
            View forecast predictions and compare with historical data.
          </p>
        </div>

        {/* Filters */}
        <div className="bg-white rounded-lg shadow-md p-6 mb-6">
          <div className="flex flex-wrap gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Material ID
              </label>
              <select
                value={selectedMaterial}
                onChange={(e) => setSelectedMaterial(e.target.value)}
                className="border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="">All Materials</option>
                {[...new Set(forecastData.map((f) => f.material_id))].map(
                  (material) => (
                    <option key={material} value={material}>
                      {material}
                    </option>
                  )
                )}
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Time Period
              </label>
              <select
                value={selectedPeriod}
                onChange={(e) => setSelectedPeriod(e.target.value)}
                className="border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="all">All Time</option>
                <option value="week">Next Week</option>
                <option value="month">Next Month</option>
                <option value="quarter">Next Quarter</option>
              </select>
            </div>
          </div>
        </div>

        {/* Summary Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-6">
          <div className="bg-white rounded-lg shadow-md p-6">
            <h3 className="text-lg font-semibold text-gray-900">
              Total Materials
            </h3>
            <p className="text-3xl font-bold text-blue-600">
              {summaryData.total_materials || 0}
            </p>
          </div>
          <div className="bg-white rounded-lg shadow-md p-6">
            <h3 className="text-lg font-semibold text-gray-900">
              Avg Forecast P50
            </h3>
            <p className="text-3xl font-bold text-green-600">
              {summaryData.average_forecast?.p50?.toFixed(1) || "0"}
            </p>
          </div>
          <div className="bg-white rounded-lg shadow-md p-6">
            <h3 className="text-lg font-semibold text-gray-900">
              Forecast Volatility
            </h3>
            <p className="text-3xl font-bold text-orange-600">
              {(summaryData.forecast_volatility * 100)?.toFixed(1) || "0"}%
            </p>
          </div>
          <div className="bg-white rounded-lg shadow-md p-6">
            <h3 className="text-lg font-semibold text-gray-900">
              Materials Needing Orders
            </h3>
            <p className="text-3xl font-bold text-red-600">
              {summaryData.procurement?.materials_needing_orders || 0}
            </p>
          </div>
        </div>

        {/* Charts */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
          {/* Forecast vs Historical Chart */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">
              Forecast vs Historical Demand
            </h3>
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Area
                  type="monotone"
                  dataKey="historical"
                  stackId="1"
                  stroke="#8884d8"
                  fill="#8884d8"
                  name="Historical"
                />
                <Area
                  type="monotone"
                  dataKey="forecast_p50"
                  stackId="2"
                  stroke="#82ca9d"
                  fill="#82ca9d"
                  name="Forecast P50"
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>

          {/* Forecast Range Chart */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">
              Forecast Ranges by Material
            </h3>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={summaryChartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="material" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="p10" fill="#8884d8" name="P10" />
                <Bar dataKey="p50" fill="#82ca9d" name="P50" />
                <Bar dataKey="p90" fill="#ffc658" name="P90" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Procurement Metrics Table */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            Procurement Recommendations
          </h3>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Material
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Current Stock
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Safety Stock
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Reorder Point
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Recommended Order
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Recommendations
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {forecastData.map((forecast) => (
                  <tr key={forecast.material_id}>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                      {forecast.material_id}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {forecast.current_stock?.toFixed(1) || "N/A"}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {forecast.safety_stock?.toFixed(1) || "N/A"}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {forecast.reorder_point?.toFixed(1) || "N/A"}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {forecast.recommended_order_qty?.toFixed(1) || "N/A"}
                    </td>
                    <td className="px-6 py-4 text-sm text-gray-500">
                      <div className="space-y-1">
                        {forecast.recommendations?.map((rec, index) => (
                          <div
                            key={index}
                            className={`text-xs px-2 py-1 rounded ${
                              rec.type === "critical"
                                ? "bg-red-100 text-red-800"
                                : rec.type === "urgent"
                                ? "bg-orange-100 text-orange-800"
                                : "bg-blue-100 text-blue-800"
                            }`}
                          >
                            {rec.message}
                          </div>
                        ))}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ForecastDashboard;
