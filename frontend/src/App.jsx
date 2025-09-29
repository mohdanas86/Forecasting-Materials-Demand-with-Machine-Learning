import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Navigation from "./components/Navigation";
import UploadData from "./pages/UploadData";
import ForecastDashboard from "./pages/ForecastDashboard";
import Alerts from "./pages/Alerts";

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-gray-50">
        <Navigation />
        <Routes>
          <Route path="/" element={<UploadData />} />
          <Route path="/forecast" element={<ForecastDashboard />} />
          <Route path="/alerts" element={<Alerts />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
