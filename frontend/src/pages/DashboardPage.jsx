import { useContext, useMemo, useState } from "react";
import {
  Activity,
  ArrowUpRight,
  Camera,
  CheckCircle2,
  Cpu,
  Eye,
  FileText,
  Info,
  LogIn,
  LogOut,
  MessageSquare,
  Search,
  ShieldAlert,
  UserCheck,
  UserPlus,
  Users
} from "lucide-react";
import { SurveillanceContext } from "../context/SurveillanceContext";
import DetailModal from "../components/DetailModal";
import {
  describeEvent,
  formatEventStamp,
  humanizeSeconds,
  shortSessionId
} from "../lib/eventDisplay";
import { buildSessionInsights } from "../lib/sessionInsights";
import "./dashboard.css";

// Icons are keyed by the stable `key` the humanizer returns, so adding a new
// event type never means touching the table markup.
const EVENT_ICONS = {
  "person-in": LogIn,
  "person-out": LogOut,
  "attention-start": Eye,
  "attention-end": Eye,
  unknown: ShieldAlert,
  snapshot: Camera,
  chat: MessageSquare,
  "chat-request": MessageSquare,
  "action-done": CheckCircle2,
  session: Activity,
  enroll: UserPlus,
  summary: FileText,
  memory: Search,
  train: Cpu,
  generic: Info
};

const CARD_META = [
  { id: "detections", label: "Total Detections", tone: "blue", icon: Activity, hint: "All detections, known people and unknown alerts" },
  { id: "known", label: "Known People", tone: "green", icon: UserCheck, hint: "Who was seen, and how long they held attention" },
  { id: "unknown", label: "Unknown Alerts", tone: "red", icon: ShieldAlert, hint: "Unrecognised faces and how often they appear" }
];

function Stat({ label, value, tone }) {
  return (
    <div className={`detail-stat${tone ? ` ${tone}` : ""}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function DetectionsSection({ insights }) {
  const { live, sessions, detectionTotals, objectClasses } = insights;
  return (
    <>
      <div className="detail-section">
        <h4>Right now</h4>
        <div className="detail-stat-row">
          <Stat label="Faces in frame" value={live.faces} />
          <Stat label="Objects in frame" value={live.objects} />
          <Stat label="People tracked this session" value={live.peopleTracked} />
          <Stat
            label="Unknown alerts this session"
            value={live.unknownAlerts}
            tone={live.unknownAlerts > 0 ? "alert" : undefined}
          />
        </div>
        {live.available ? (
          <div className="chip-row" style={{ marginTop: 10 }}>
            {live.activeSubjects.length ? (
              live.activeSubjects.map((subject) => (
                <span key={subject.name} className="chip known">
                  {subject.name} · {Math.round(subject.confidence * 100)}%
                </span>
              ))
            ) : (
              <span className="chip muted">No recognised face in the current frame</span>
            )}
            {live.activeObjects.map((label) => (
              <span key={label} className="chip muted">
                {label}
              </span>
            ))}
          </div>
        ) : (
          <p className="detail-note">The pipeline is idle — showing recorded session history below.</p>
        )}
      </div>

      <div className="detail-section">
        <h4>Recorded history</h4>
        <div className="detail-stat-row">
          <Stat label="Sessions recorded" value={detectionTotals.sessions} />
          <Stat label="Detections logged" value={detectionTotals.detections} />
          <Stat label="Objects logged" value={detectionTotals.objects} />
          <Stat label="Monitored time" value={humanizeSeconds(detectionTotals.monitoredSec)} />
        </div>
      </div>

      <div className="detail-section">
        <h4>Per session</h4>
        {sessions.length ? (
          <table className="detail-table">
            <thead>
              <tr>
                <th>Started (UTC)</th>
                <th>Duration</th>
                <th>Frames</th>
                <th>Detections</th>
                <th>Known</th>
                <th>Unknown</th>
                <th>People</th>
              </tr>
            </thead>
            <tbody>
              {sessions.map((session) => (
                <tr key={`${session.id}-${session.timestampUtc}`}>
                  <td data-label="Started">{formatEventStamp(session.timestampUtc)}</td>
                  <td data-label="Duration">{humanizeSeconds(session.durationSec)}</td>
                  <td data-label="Frames">{session.frames}</td>
                  <td data-label="Detections">{session.detections}</td>
                  <td data-label="Known">{session.knownFrames}</td>
                  <td data-label="Unknown">{session.unknownAlerts}</td>
                  <td data-label="People">
                    {session.subjects.length
                      ? session.subjects.map((subject) => subject.name).join(", ")
                      : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p className="detail-empty">No monitoring sessions recorded yet.</p>
        )}
      </div>

      {objectClasses.length ? (
        <div className="detail-section">
          <h4>Objects detected</h4>
          <div className="chip-row">
            {objectClasses.map((row) => (
              <span key={row.label} className="chip muted">
                {row.label} · {row.count}
              </span>
            ))}
          </div>
        </div>
      ) : null}
    </>
  );
}

function KnownPeopleSection({ insights }) {
  const { knownPeople } = insights;
  if (!knownPeople.length) {
    return <p className="detail-empty">No known person has been recorded yet. Enroll a face to start naming people.</p>;
  }
  return (
    <div className="detail-section">
      <h4>Known people</h4>
      <table className="detail-table">
        <thead>
          <tr>
            <th>Person</th>
            <th>Best match</th>
            <th>Sessions</th>
            <th>Attention time</th>
            <th>Last seen (UTC)</th>
          </tr>
        </thead>
        <tbody>
          {knownPeople.map((person) => (
            <tr key={person.name}>
              <td data-label="Person">
                {person.name}
                {person.currentlyVisible ? <span className="chip known chip-now"> visible now</span> : null}
                {person.enrolled ? <span className="chip muted"> enrolled</span> : null}
              </td>
              <td data-label="Best match">
                {person.bestConfidence > 0 ? `${Math.round(person.bestConfidence * 100)}%` : "—"}
              </td>
              <td data-label="Sessions">{person.sessionsSeen}</td>
              <td data-label="Attention time">
                {person.attentionSec > 0 ? humanizeSeconds(person.attentionSec) : "—"}
              </td>
              <td data-label="Last seen">{person.lastSeenUtc ? formatEventStamp(person.lastSeenUtc) : "—"}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="detail-note">
        Attention time is how long the behaviour tracker had this person focused on a detected object.
      </p>
    </div>
  );
}

function UnknownAlertsSection({ insights }) {
  const { unknownAlerts } = insights;
  return (
    <>
      <div className="detail-section">
        <h4>Alert summary</h4>
        <div className="detail-stat-row">
          <Stat
            label="Unknown alerts"
            value={unknownAlerts.total}
            tone={unknownAlerts.total > 0 ? "alert" : undefined}
          />
          <Stat label="Average per monitored minute" value={unknownAlerts.densityPerMin} />
          <Stat label="Frames with unknown faces" value={unknownAlerts.unknownFrames} />
          <Stat
            label="Sessions with alerts"
            value={`${unknownAlerts.sessionsWithAlerts}/${unknownAlerts.totalSessions}`}
          />
        </div>
      </div>

      <div className="detail-section">
        <h4>Sessions that raised alerts</h4>
        {unknownAlerts.sessions.length ? (
          <table className="detail-table">
            <thead>
              <tr>
                <th>Session ended (UTC)</th>
                <th>Alerts</th>
                <th>Frames with unknown faces</th>
                <th>Per minute</th>
              </tr>
            </thead>
            <tbody>
              {unknownAlerts.sessions.map((session) => (
                <tr key={`${session.id}-${session.timestampUtc}`}>
                  <td data-label="Session">{formatEventStamp(session.timestampUtc)}</td>
                  <td data-label="Alerts">{session.unknownAlerts}</td>
                  <td data-label="Frames">{session.unknownFrames}</td>
                  <td data-label="Per minute">{session.unknownDensityPerMin}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p className="detail-empty">No unknown face has been detected in the recorded sessions.</p>
        )}
        <p className="detail-note">
          An alert is raised at most once per few seconds per unknown face, so the count reads as “distinct
          strangers”, not frames. Captured images are written to the machine running the pipeline
          (<code>unknown_incidents/</code>) and are never uploaded.
        </p>
      </div>
    </>
  );
}

function DashboardPage() {
  const { logs, detections, behavior, status } = useContext(SurveillanceContext);
  const [detail, setDetail] = useState({ open: false, tab: "detections" });

  const insights = useMemo(
    () => buildSessionInsights({ logs, detections, behavior }),
    [logs, detections, behavior]
  );

  const cardValues = {
    detections: insights.totals.total,
    known: insights.totals.known,
    unknown: insights.totals.unknown
  };

  const recentRows = useMemo(() => {
    const rows = [...(logs || [])].reverse();
    return rows.slice(0, 25).map((row) => ({ row, display: describeEvent(row) }));
  }, [logs]);

  const openDetail = (tab) => setDetail({ open: true, tab });

  return (
    <div className="page-content">
      <div className="page-header">
        <h1 className="page-title">Intelligence Dashboard</h1>
        <p className="page-subtitle">Global monitoring session state, alerts, and event telemetry.</p>
      </div>

      <div className="dash-grid">
        {CARD_META.map((card) => {
          const Icon = card.icon;
          return (
            <button
              key={card.id}
              type="button"
              className="glass-panel-interactive metric-card"
              onClick={() => openDetail(card.id)}
              aria-haspopup="dialog"
              title={card.hint}
            >
              <div className={`metric-icon ${card.tone}`}>
                <Icon size={24} />
              </div>
              <div className="metric-copy">
                <h4>{card.label}</h4>
                <div>{cardValues[card.id]}</div>
                <span className="metric-hint">
                  {insights.totals.source === "live" ? "Live session" : "Last recorded session"} · view details
                </span>
              </div>
              <ArrowUpRight className="metric-open" size={18} aria-hidden="true" />
            </button>
          );
        })}
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
                <th>When (UTC)</th>
                <th>Event</th>
                <th>What happened</th>
              </tr>
            </thead>
            <tbody>
              {recentRows.length ? (
                recentRows.map(({ row, display }, idx) => {
                  const Icon = EVENT_ICONS[display.key] || EVENT_ICONS.generic;
                  return (
                    <tr key={`${row.timestamp_utc || "na"}-${idx}`}>
                      <td className="event-when">{formatEventStamp(row.timestamp_utc)}</td>
                      <td>
                        <span className={`event-title tone-${display.tone}`}>
                          <Icon size={15} aria-hidden="true" />
                          {display.title}
                        </span>
                        {/* Machine-facing metadata sits together, below the human line. */}
                        <span className="event-meta">
                          <span className="event-type-badge">{display.rawType}</span>
                          {row.session_id ? (
                            <span className="event-session" title={row.session_id}>
                              {shortSessionId(row.session_id)}
                            </span>
                          ) : null}
                        </span>
                      </td>
                      <td>{display.summary}</td>
                    </tr>
                  );
                })
              ) : (
                <tr>
                  <td colSpan={3} className="empty-cell">
                    No events yet.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      <DetailModal
        open={detail.open}
        title="Session detections"
        subtitle={
          insights.lastUpdatedUtc
            ? `Latest recorded activity: ${formatEventStamp(insights.lastUpdatedUtc)}`
            : "No recorded activity yet"
        }
        tabs={[
          { id: "detections", label: "Detections", count: insights.sessionCount },
          { id: "known", label: "Known people", count: insights.knownPeople.length },
          { id: "unknown", label: "Unknown alerts", count: insights.unknownAlerts.total }
        ]}
        activeTab={detail.tab}
        onTabChange={(tab) => setDetail((prev) => ({ ...prev, tab }))}
        onClose={() => setDetail((prev) => ({ ...prev, open: false }))}
      >
        {detail.tab === "detections" ? <DetectionsSection insights={insights} /> : null}
        {detail.tab === "known" ? <KnownPeopleSection insights={insights} /> : null}
        {detail.tab === "unknown" ? <UnknownAlertsSection insights={insights} /> : null}
      </DetailModal>
    </div>
  );
}

export default DashboardPage;
