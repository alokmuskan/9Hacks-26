import { useContext, useState } from "react";
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
  const [objectQuery, setObjectQuery] = useState("");
  const [personQuery, setPersonQuery] = useState("");
  const [textQuery, setTextQuery] = useState("");
  const [result, setResult] = useState(null);
  const [searchRows, setSearchRows] = useState([]);
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
          <button className="btn-secondary" onClick={() => run(refreshMemoryStats)} disabled={busy}>
            Refresh
          </button>
        </section>

        <section className="glass-panel">
          <h3>Recent Snapshots</h3>
          <button
            className="btn-secondary"
            disabled={busy}
            onClick={() => run(async () => setRecentRows((await api.memoryRecent(10, 40)).items || []))}
          >
            Load Last 10 Minutes
          </button>
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
