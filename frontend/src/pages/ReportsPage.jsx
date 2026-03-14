import { useContext, useState } from "react";
import { FileText } from "lucide-react";
import { api } from "../api/client";
import { SurveillanceContext } from "../context/SurveillanceContext";
import "./reports.css";

function ReportsPage() {
  const { lastSummary, runSessionSummary, logs, refreshLogs, behavior } = useContext(SurveillanceContext);
  const [minutes, setMinutes] = useState(5);
  const [summaryText, setSummaryText] = useState("");
  const [busy, setBusy] = useState(false);

  const run = async (fn) => {
    try {
      setBusy(true);
      await fn();
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="page-content">
      <div className="page-header">
        <h1 className="page-title">Summaries and Reports</h1>
        <p className="page-subtitle">Situation summaries and activity report telemetry.</p>
      </div>

      <div className="report-grid">
        <section className="glass-panel">
          <h3>Situation Summary</h3>
          <div className="summary-controls">
            <div className="field">
              <label>Lookback Minutes</label>
              <input
                type="number"
                min={1}
                max={1440}
                value={minutes}
                onChange={(e) => setMinutes(Number(e.target.value))}
              />
            </div>
            <button
              className="btn-primary"
              disabled={busy}
              onClick={() =>
                run(async () => {
                  const row = await runSessionSummary(minutes);
                  setSummaryText(row.rendered || "");
                })
              }
            >
              <FileText size={16} />
              Generate
            </button>
            <button className="btn-secondary" disabled={busy} onClick={() => run(refreshLogs)}>
              Refresh Logs
            </button>
          </div>
          <pre className="text-box">{summaryText || "Generate a summary to view narrative output."}</pre>
          <pre className="json-box">{JSON.stringify(lastSummary || {}, null, 2)}</pre>
        </section>

        <section className="glass-panel">
          <h3>Behavior Snapshot</h3>
          <pre className="json-box">{JSON.stringify(behavior || {}, null, 2)}</pre>
        </section>
      </div>

      <section className="glass-panel" style={{ marginTop: 16 }}>
        <h3>Recent Log Rows</h3>
        <div className="report-log-table">
          <table className="log-table">
            <thead>
              <tr>
                <th>Time</th>
                <th>Type</th>
                <th>Session</th>
                <th>Details</th>
              </tr>
            </thead>
            <tbody>
              {[...logs].slice(-60).reverse().map((row, idx) => (
                <tr key={`${row.timestamp_utc}-${idx}`}>
                  <td>{row.timestamp_utc || "-"}</td>
                  <td>{row.event_type || row.type || "-"}</td>
                  <td>{row.session_id || "-"}</td>
                  <td>{row.message || row.intent || row.event || "-"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}

export default ReportsPage;
