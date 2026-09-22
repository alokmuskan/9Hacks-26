import { useContext } from "react";
import { Navigate, Route, Routes } from "react-router-dom";
import LoadingScreen from "./components/LoadingScreen";
import Sidebar from "./components/Sidebar";
import { SurveillanceContext, SurveillanceProvider } from "./context/SurveillanceContext";
import DashboardPage from "./pages/DashboardPage";
import LiveMonitorPage from "./pages/LiveMonitorPage";
import MemoryPage from "./pages/MemoryPage";
import ReportsPage from "./pages/ReportsPage";
import ChatPage from "./pages/ChatPage";
import RegisterPage from "./pages/RegisterPage";
import "./App.css";

function AppShell() {
  const { initializing, operation, errors, removeError } = useContext(SurveillanceContext);

  const appBusy = Boolean(initializing || operation?.active);
  const visibleErrors = (errors || []).slice(0, 3);

  return (
    <>
      <div className="app-container" aria-busy={appBusy}>
        <Sidebar />
        <Routes>
          <Route path="/" element={<DashboardPage />} />
          <Route path="/live" element={<LiveMonitorPage />} />
          <Route path="/memory" element={<MemoryPage />} />
          <Route path="/reports" element={<ReportsPage />} />
          <Route path="/chat" element={<ChatPage />} />
          <Route path="/register" element={<RegisterPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </div>

      <LoadingScreen
        visible={Boolean(initializing)}
        title="Connecting to backend"
        message="Loading realtime status, memory, and logs..."
      />
      <LoadingScreen
        visible={Boolean(!initializing && operation?.active)}
        title="Please wait"
        message={operation?.message || "Working..."}
      />

      {visibleErrors.length > 0 && (
        <div className="error-stack" aria-live="polite">
          {visibleErrors.map((row) => (
            <div key={row.id} className="error-toast">
              <div className="error-toast-title">Request failed</div>
              <div className="error-toast-message">{row.message}</div>
              <button className="error-toast-close" onClick={() => removeError(row.id)}>
                Dismiss
              </button>
            </div>
          ))}
        </div>
      )}
    </>
  );
}

function App() {
  return (
    <SurveillanceProvider>
      <AppShell />
    </SurveillanceProvider>
  );
}

export default App;
