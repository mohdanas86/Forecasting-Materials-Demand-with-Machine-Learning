import { useState } from "react";
import { apiService } from "../services/api";

const UploadData = () => {
  const [uploadStatus, setUploadStatus] = useState({});
  const [isUploading, setIsUploading] = useState({});

  const uploadConfigs = [
    {
      key: "projects",
      title: "Projects Data",
      description: "Upload project information including timelines and budgets",
      endpoint: "uploadProjects",
      accept: ".csv,.json",
    },
    {
      key: "materials",
      title: "Materials Data",
      description: "Upload material master data with specifications and costs",
      endpoint: "uploadMaterials",
      accept: ".csv,.json",
    },
    {
      key: "historical",
      title: "Historical Demand",
      description: "Upload historical demand data for forecasting",
      endpoint: "uploadHistorical",
      accept: ".csv,.json",
    },
    {
      key: "inventory",
      title: "Current Inventory",
      description: "Upload current inventory levels and stock information",
      endpoint: "uploadInventory",
      accept: ".csv,.json",
    },
  ];

  const handleFileUpload = async (file, config) => {
    if (!file) return;

    setIsUploading((prev) => ({ ...prev, [config.key]: true }));
    setUploadStatus((prev) => ({ ...prev, [config.key]: null }));

    try {
      const response = await apiService[config.endpoint](file);
      setUploadStatus((prev) => ({
        ...prev,
        [config.key]: {
          success: true,
          message: `Successfully uploaded ${response.data.rows_inserted} rows`,
          details: response.data,
        },
      }));
    } catch (error) {
      console.error("Upload error:", error);
      setUploadStatus((prev) => ({
        ...prev,
        [config.key]: {
          success: false,
          message: error.response?.data?.detail || "Upload failed",
          details: error.response?.data,
        },
      }));
    } finally {
      setIsUploading((prev) => ({ ...prev, [config.key]: false }));
    }
  };

  const UploadCard = ({ config }) => (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h3 className="text-lg font-semibold text-gray-900 mb-2">
        {config.title}
      </h3>
      <p className="text-gray-600 mb-4">{config.description}</p>

      <div className="space-y-4">
        <input
          type="file"
          accept={config.accept}
          onChange={(e) => handleFileUpload(e.target.files[0], config)}
          disabled={isUploading[config.key]}
          className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100 disabled:opacity-50"
        />

        {isUploading[config.key] && (
          <div className="flex items-center text-blue-600">
            <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600 mr-2"></div>
            Uploading...
          </div>
        )}

        {uploadStatus[config.key] && (
          <div
            className={`p-3 rounded-md ${
              uploadStatus[config.key].success
                ? "bg-green-50 text-green-800"
                : "bg-red-50 text-red-800"
            }`}
          >
            <p className="font-medium">{uploadStatus[config.key].message}</p>
            {uploadStatus[config.key].details && (
              <div className="mt-2 text-sm">
                <p>
                  Rows received:{" "}
                  {uploadStatus[config.key].details.rows_received}
                </p>
                <p>
                  Rows inserted:{" "}
                  {uploadStatus[config.key].details.rows_inserted}
                </p>
                {uploadStatus[config.key].details.rows_failed > 0 && (
                  <p>
                    Rows failed: {uploadStatus[config.key].details.rows_failed}
                  </p>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900">Upload Data</h1>
          <p className="mt-2 text-gray-600">
            Upload your project, material, historical demand, and inventory data
            to get started with forecasting.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {uploadConfigs.map((config) => (
            <UploadCard key={config.key} config={config} />
          ))}
        </div>

        <div className="mt-8 bg-blue-50 border border-blue-200 rounded-lg p-4">
          <h3 className="text-lg font-medium text-blue-900 mb-2">
            File Format Guidelines
          </h3>
          <ul className="text-blue-800 text-sm space-y-1">
            <li>• Supported formats: CSV and JSON</li>
            <li>• CSV files should have headers in the first row</li>
            <li>• JSON files should contain an array of objects</li>
            <li>• Date fields should be in ISO format (YYYY-MM-DD)</li>
            <li>• Numeric fields will be automatically converted</li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default UploadData;
