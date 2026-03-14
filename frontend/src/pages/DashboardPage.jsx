import { useContext, useMemo } from "react";
import { Activity, ShieldAlert, UserCheck, Users } from "lucide-react";
import { SurveillanceContext } from "../context/SurveillanceContext";
import "./dashboard.css";

function DashboardPage() {
  const { logs, detections, status } = useContext(SurveillanceContext);

  const metrics = useMemo(() => {
    const hasLiveCounts =
      Boolean(detections?.counts) &&
      ("known_detections" in detections.counts || "unknown_detections" in detections.counts);

    if (hasLiveCounts) {
      const known = Number(detections?.counts?.known_detections || 0);
      const unknown = Number(detections?.counts?.unknown_detections || 0);
      return {
        known,
        unknown,
        total: known + unknown
      };
    }

    const latestSession = [...logs].reverse().find((row) => row?.event_type === "recognize_session") || null;
    const aggregate = latestSession?.aggregate && typeof latestSession.aggregate === "object" ? latestSession.aggregate : {};
    const known = Number(aggregate?.known_detections ?? latestSession?.known_detections ?? 0);
    const unknown = Number(aggregate?.unknown_detections ?? latestSession?.unknown_detections ?? 0);

    return {
      known,
      unknown,
      total: known + unknown
    };
  }, [detections, logs]);

  const recentRows = useMemo(() => {
    const rows = [...logs].reverse();
    return rows.slice(0, 25);
  }, [logs]);

  return (
    <div className="page-content">
      <div className="page-header">
        <h1 className="page-title">Intelligence Dashboard</h1>
        <p className="page-subtitle">Global monitoring session state, alerts, and event telemetry.</p>
      </div>

      <div className="dash-grid">
        <div className="glass-panel-interactive metric-card">
          <div className="metric-icon blue">
            <Activity size={24} />
          </div>
          <div className="metric-copy">
            <h4>Total Detections</h4>
            <div>{metrics.total}</div>
          </div>
        </div>
        <div className="glass-panel-interactive metric-card">
          <div className="metric-icon green">
            <UserCheck size={24} />
          </div>
          <div className="metric-copy">
            <h4>Known Faces</h4>
            <div>{metrics.known}</div>
          </div>
        </div>
        <div className="glass-panel-interactive metric-card">
          <div className="metric-icon red">
            <ShieldAlert size={24} />
          </div>
          <div className="metric-copy">
            <h4>Unknown Faces</h4>
            <div>{metrics.unknown}</div>
          </div>
        </div>
      </div>

      <div className="glass-panel" style={{ marginTop: 16 }}>
        <div className="table-head">
          <h3>Recent Events</h3>
          <span className="tiny">
            {status?.running ? "Pipeline running" : "Pipeline idle"} · {status?.mode || "idle"}
          </span>
          <Users size={16} color="var(--text-muted)" />
        </div>
        <div className="table-wrap">
          <table className="log-table">
            <thead>
              <tr>
                <th>UTC Time</th>
                <th>Type</th>
                <th>Session / Subject</th>
                <th>Details</th>
              </tr>
            </thead>
            <tbody>
              {recentRows.length ? (
                recentRows.map((row, idx) => (
                  <tr key={`${row.timestamp_utc || "na"}-${idx}`}>
                    <td>{row.timestamp_utc || "-"}</td>
                    <td>{row.event_type || row.type || "-"}</td>
                    <td>{row.session_id || row.person || "-"}</td>
                    <td>{row.message || row.intent || row.event || "-"}</td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan={4} className="empty-cell">
                    No events yet.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

export default DashboardPage;
