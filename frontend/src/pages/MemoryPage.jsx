import { useContext, useEffect, useMemo, useState } from "react";
import { Search } from "lucide-react";
import { api } from "../api/client";
import { SurveillanceContext } from "../context/SurveillanceContext";
import "./memory.css";

function SnapshotList({ rows }) {
  if (!rows?.length) {
    return <div className="empty-cell">No snapshots found.</div>;
  }
  return (
    <div className="snapshot-grid">
      {rows.map((row) => (
        <div className="snapshot-card" key={row.id || row.snapshot || row.timestamp_utc}>
          <div className="tiny">{row.timestamp_local || row.timestamp_utc}</div>
          <div className="muted">{(row.objects || []).join(", ") || "No objects"}</div>
          <div className="tiny">{row.snapshot}</div>
        </div>
      ))}
    </div>
  );
}

function MemoryPage() {
  const { memoryStats, refreshMemoryStats } = useContext(SurveillanceContext);
  const [recentRows, setRecentRows] = useState([]);
  const [recentMeta, setRecentMeta] = useState(null);
  const [objectQuery, setObjectQuery] = useState("");
  const [personQuery, setPersonQuery] = useState("");
  const [textQuery, setTextQuery] = useState("");
  const [result, setResult] = useState(null);
  const [searchRows, setSearchRows] = useState([]);
  const [busy, setBusy] = useState(false);
  const [statsFlash, setStatsFlash] = useState(false);

  const run = async (fn) => {
    try {
      setBusy(true);
      await fn();
    } finally {
      setBusy(false);
    }
  };

  // Ask the backend when the last monitoring session ended; the picker offers
  // windows up to that point only, so the user cannot request a range that is
  // entirely before any session ran. Falls back to 60 min when the backend is
  // unreachable or no session has ever been recorded.
  const [maxWindowMinutes, setMaxWindowMinutes] = useState(60);
  useEffect(() => {
    let cancelled = false;
    api
      .sessionWindow()
      .then((row) => {
        if (!cancelled && Number.isFinite(row?.session_window_minutes)) {
          setMaxWindowMinutes(Math.max(1, row.session_window_minutes));
        }
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, []);
  const windowOptions = useMemo(
    () =>
      [1, 5, 15, 30, 60, 180, 720, 1440]
        .map((m) => ({
          minutes: m,
          label: m >= 60 ? `${m / 60} h` : `${m} min`,
        }))
        .filter((o) => o.minutes <= maxWindowMinutes),
    [maxWindowMinutes]
  );
  const [windowMinutes, setWindowMinutes] = useState(10);
  useEffect(() => {
    if (windowOptions.length && !windowOptions.some((o) => o.minutes === windowMinutes)) {
      setWindowMinutes(windowOptions[windowOptions.length - 1].minutes);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [windowOptions]);

  const loadRecent = async (minutes = windowMinutes) => {
    const row = await api.memoryRecent(minutes, 40);
    setRecentRows(row.items || []);
    setRecentMeta(row);
  };

  const refreshStats = async () => {
    await refreshMemoryStats();
    setStatsFlash(true);
    setTimeout(() => setStatsFlash(false), 1200);
  };

  return (
    <div className="page-content">
      <div className="page-header">
        <h1 className="page-title">Memory</h1>
        <p className="page-subtitle">Snapshots, last-seen lookups, and searchable history.</p>
      </div>

      <div className="memory-grid">
        <section className="glass-panel">
          <h3>Memory Stats</h3>
          <div className="tiny" style={{ marginBottom: 12 }}>
            Snapshot storage and vector backend status.
          </div>
          <pre className="json-box">{JSON.stringify(memoryStats || {}, null, 2)}</pre>
          <button
            className="btn-secondary"
            onClick={() => run(refreshStats)}
            disabled={busy}
            title="Re-read snapshot storage and vector backend status from the backend"
          >
            {statsFlash ? "Updated ✓" : busy ? "Refreshing…" : "Refresh"}
          </button>
          {statsFlash && <span className="tiny muted" style={{ marginLeft: 8 }}>Memory stats are up to date</span>}
        </section>

        <section className="glass-panel">
          <h3>Recent Snapshots</h3>
          <div className="lookup-row">
            <div className="field">
              <label>Time window</label>
              <select
                value={windowMinutes}
                onChange={(e) => setWindowMinutes(Number(e.target.value))}
                disabled={busy || !windowOptions.length}
              >
                {windowOptions.map((o) => (
                  <option key={o.minutes} value={o.minutes}>
                    Last {o.label}
                  </option>
                ))
                }
              </select>
            </div>
            <button className="btn-secondary" disabled={busy || !windowOptions.length} onClick={() => run(() => loadRecent())}>
              Load
            </button>
          </div>
          {recentMeta?.capped && (
            <p className="tiny muted" style={{ margin: "6px 0 0" }}>
              Capped to the last monitoring session ({recentMeta.session_window_minutes} min window).
            </p>
          )}
          <SnapshotList rows={recentRows} />
        </section>
      </div>

      <div className="memory-grid" style={{ marginTop: 16 }}>
        <section className="glass-panel">
          <h3>Last Seen Lookup</h3>
          <div className="lookup-row">
            <div className="field">
              <label>Object</label>
              <input value={objectQuery} onChange={(e) => setObjectQuery(e.target.value)} placeholder="laptop" />
            </div>
            <button
              className="btn-secondary"
              disabled={!objectQuery.trim() || busy}
              onClick={() =>
                run(async () => {
                  const row = await api.memoryFindObject(objectQuery.trim());
                  setResult(row.item || null);
                })
              }
            >
              Find Object
            </button>
          </div>
          <div className="lookup-row">
            <div className="field">
              <label>Person</label>
              <input value={personQuery} onChange={(e) => setPersonQuery(e.target.value)} placeholder="Your Name" />
            </div>
            <button
              className="btn-secondary"
              disabled={!personQuery.trim() || busy}
              onClick={() =>
                run(async () => {
                  const row = await api.memoryFindPerson(personQuery.trim());
                  setResult(row.item || null);
                })
              }
            >
              Find Person
            </button>
          </div>
          <pre className="json-box">{JSON.stringify(result || {}, null, 2)}</pre>
        </section>

        <section className="glass-panel">
          <h3>Search History</h3>
          <div className="lookup-row">
            <div className="field" style={{ flex: 1 }}>
              <label>Text Query</label>
              <input
                value={textQuery}
                onChange={(e) => setTextQuery(e.target.value)}
                placeholder="person using laptop"
              />
            </div>
            <button
              className="btn-primary"
              disabled={!textQuery.trim() || busy}
              onClick={() => run(async () => setSearchRows((await api.memorySearch(textQuery.trim(), 10)).items || []))}
            >
              <Search size={16} />
              Search
            </button>
          </div>
          <SnapshotList rows={searchRows} />
        </section>
      </div>
    </div>
  );
}

export default MemoryPage;
