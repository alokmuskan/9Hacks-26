// The metrics log is written for machines: one row per backend event, technical
// field names, no prose. The dashboard shows that log to people, so this module
// is the translation layer — every known event type gets a friendly title, a
// plain sentence, and a tone the UI can colour. Unknown types degrade to a
// prettified type name instead of leaking raw snake_case into the feed.
//
// Kept dependency-free and pure so it can be unit-tested without a DOM.
// Source of truth for the shapes: server.py / main.py `_append_metric` calls.

const INTENT_LABELS = {
  greeting: "greeting",
  open_ended: "open question",
  person_count: "people count",
  person_last_seen: "person lookup",
  object_last_seen: "object lookup",
  last_seen: "last seen",
  presence: "presence check",
  attention_lookup: "attention lookup",
  session_summary: "session summary",
  memory_recent: "recent activity",
  memory_stats: "memory stats",
  action_proposal: "action request",
  action_confirm: "action confirmation",
  action_execute: "action"
};

const ACTION_LABELS = {
  monitor_start: "Start monitoring",
  monitor_stop: "Stop monitoring",
  run_summary: "Run a session summary",
  snapshot: "Save a snapshot",
  toggle: "Change detection toggles",
  toggle_general_yolo: "Toggle general YOLO",
  toggle_custom_yolo: "Toggle custom YOLO",
  toggle_gaze: "Toggle gaze tracking",
  confirm_invalid: "Rejected an expired confirmation",
  none: "No action"
};

function num(value, fallback = 0) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function plural(count, singular, pluralForm) {
  return `${count} ${count === 1 ? singular : pluralForm || `${singular}s`}`;
}

export function humanizeSeconds(value) {
  const total = Math.max(Math.round(num(value, 0)), 0);
  if (total < 60) {
    return `${total}s`;
  }
  const minutes = Math.floor(total / 60);
  const seconds = total % 60;
  return seconds ? `${minutes}m ${seconds}s` : `${minutes}m`;
}

export function prettifyEventType(type) {
  const text = String(type || "").trim();
  if (!text) {
    return "Event";
  }
  const spaced = text.replace(/[_-]+/g, " ");
  return spaced.charAt(0).toUpperCase() + spaced.slice(1);
}

/**
 * Session ids carry a type prefix (`monitor-20260920-104542`); the stamp after it
 * is the part a reader can use, so drop the prefix and keep the date and clock.
 */
export function shortSessionId(id) {
  const text = String(id || "").trim();
  if (!text) {
    return "";
  }
  const parts = text.split("-");
  return parts.length > 1 ? parts.slice(1).join("-") : text;
}

export function formatEventClock(iso) {
  const ts = Date.parse(String(iso || ""));
  if (Number.isNaN(ts)) {
    return String(iso || "—");
  }
  return new Date(ts).toLocaleTimeString([], { hour12: false });
}

export function formatEventStamp(iso) {
  const ts = Date.parse(String(iso || ""));
  if (Number.isNaN(ts)) {
    return String(iso || "—");
  }
  const date = new Date(ts);
  const day = date.getDate();
  const month = date.toLocaleString([], { month: "short" });
  return `${day} ${month}, ${date.toLocaleTimeString([], { hour12: false })}`;
}

function describeBehaviorEvent(row) {
  const who = String(row.person || row.subject || "Someone").trim() || "Someone";
  const target = String(row.target_object || row.object || "").trim();
  const kind = String(row.event || row.action || "").toLowerCase();
  const duration = num(row.duration_sec, 0);

  if (kind === "start") {
    return {
      key: target && target !== "person" ? "attention-start" : "person-in",
      title: target && target !== "person" ? `${who} — focus started` : `${who} entered view`,
      summary:
        target && target !== "person"
          ? `Started attending to the ${target}.`
          : "Appeared in frame and is being tracked.",
      tone: "info"
    };
  }
  if (kind === "end") {
    return {
      key: target && target !== "person" ? "attention-end" : "person-out",
      title: target && target !== "person" ? `${who} — focus ended` : `${who} left view`,
      summary:
        target && target !== "person"
          ? `Attended to the ${target} for ${humanizeSeconds(duration)}.`
          : `Was present for ${humanizeSeconds(duration)}.`,
      tone: "muted"
    };
  }
  if (kind === "transition") {
    return {
      key: "attention-start",
      title: `${who} changed focus`,
      summary: target
        ? `Moved attention to the ${target}${row.previous_target ? ` (was ${row.previous_target})` : ""}.`
        : "Attention moved to a different object.",
      tone: "info"
    };
  }
  return {
    key: "attention-start",
    title: `${who} — behaviour update`,
    summary: target ? `Now associated with the ${target}.` : "Tracking state changed.",
    tone: "info"
  };
}

function describeChatQuery(row) {
  const question = String(row.question || row.message || "").trim();
  const intent = INTENT_LABELS[String(row.intent || "")] || prettifyEventType(row.intent || "").toLowerCase();
  let source;
  if (row.used_llm) {
    source = "answered by the LLM over recorded evidence";
  } else if (row.hit === false) {
    source = "no matching recorded data";
  } else {
    source = "answered straight from recorded data";
  }
  const action = String(row.action || "none");
  const actionNote = action && action !== "none" ? ` · ${ACTION_LABELS[action] || prettifyEventType(action)}` : "";
  return {
    key: "chat",
    title: "Question asked",
    summary: `${question ? `“${question}”` : "Empty query"} · ${intent || "unclassified"}${actionNote} · ${source}.`,
    tone: row.hit === false ? "warn" : "info"
  };
}

function describeChatAction(row, phase) {
  const action = String(row.action || row.requested_action || "none");
  const label = ACTION_LABELS[action] || prettifyEventType(action);
  return {
    key: phase === "proposed" ? "chat-request" : "action-done",
    title: phase === "proposed" ? "Action proposed by chat" : "Action executed",
    summary:
      phase === "proposed"
        ? `${label} — waiting for the operator to confirm.`
        : `${label} was carried out.`,
    tone: phase === "proposed" ? "warn" : "success"
  };
}

function describeRecognizeSession(row) {
  const aggregate = row.aggregate && typeof row.aggregate === "object" ? row.aggregate : {};
  const detections = num(aggregate.detections_total, num(aggregate.known_detections) + num(aggregate.unknown_detections));
  const known = num(aggregate.unique_individuals_seen);
  const unknown = num(aggregate.unknown_alert_events);
  const duration = num(row.duration_sec, 0);
  return {
    key: "session",
    title: "Monitoring session ended",
    summary:
      `${humanizeSeconds(duration)} · ${plural(detections, "detection")} · ` +
      `${plural(known, "known person", "known people")} · ${plural(unknown, "unknown alert")}.`,
    tone: unknown > 0 ? "warn" : "success"
  };
}

function describeEnroll(row) {
  const name = String(row.name || "unnamed").trim() || "unnamed";
  const samples = num(row.samples_captured, num(row.total_samples_for_name));
  const duration = num(row.duration_sec, 0);
  return {
    key: "enroll",
    title: "Face enrolled",
    summary: `${name} · ${plural(samples, "sample")} captured in ${humanizeSeconds(duration)}.`,
    tone: "success"
  };
}

function describeSummaryQuery(row) {
  const minutes = num(row.minutes, 0);
  const window = minutes > 0 ? `last ${plural(minutes, "minute")}` : "requested window";
  return {
    key: "summary",
    title: "Session summary generated",
    summary: `${window} · ${row.hit === false ? "no matching activity found" : "activity found"}.`,
    tone: row.hit === false ? "muted" : "info"
  };
}

function describeMemoryQuery(row) {
  const text = String(row.text || row.query || row.term || "").trim();
  const rawHits = row.hits ?? row.results ?? row.count ?? row.hit_count;
  const hits = rawHits === undefined ? null : num(rawHits, 0);
  return {
    key: "memory",
    title: "Memory searched",
    summary: `${text ? `“${text}”` : "Empty query"}${hits === null ? "" : ` · ${plural(hits, "match", "matches")}`}.`,
    tone: "info"
  };
}

function describeObjectTrain(row) {
  const base = String(row.base_model || row.model || "custom model");
  const epochs = row.epochs === undefined ? null : num(row.epochs, 0);
  const duration = num(row.duration_sec, 0);
  const pieces = [base];
  if (epochs !== null) {
    pieces.push(plural(epochs, "epoch"));
  }
  if (duration > 0) {
    pieces.push(humanizeSeconds(duration));
  }
  return {
    key: "train",
    title: "Custom object model trained",
    summary: `${pieces.join(" · ")}.`,
    tone: "success"
  };
}

/**
 * Describe one metrics-log row for a human.
 *
 * @returns {{key: string, title: string, summary: string, tone: "info"|"success"|"warn"|"alert"|"muted", rawType: string}}
 */
export function describeEvent(row) {
  const type = String(row?.event_type || row?.type || "").trim();
  const rawType = type || "event";
  const base = { rawType };

  switch (type) {
    case "behavior_event":
      return { ...base, ...describeBehaviorEvent(row) };
    case "enter":
      return {
        ...base,
        key: "person-in",
        title: `${row.person || row.name || "Someone"} entered view`,
        summary: "Appeared in frame.",
        tone: "info"
      };
    case "exit":
      return {
        ...base,
        key: "person-out",
        title: `${row.person || row.name || "Someone"} left view`,
        summary: "No longer visible.",
        tone: "muted"
      };
    case "unknown_alert":
      return {
        ...base,
        key: "unknown",
        title: "Unknown face detected",
        summary: row.image_name ? `Capture saved as ${row.image_name}.` : "No enrolled identity matched this face.",
        tone: "alert"
      };
    case "memory_snapshot_auto":
      return {
        ...base,
        key: "snapshot",
        title: "Auto snapshot saved",
        summary: row.image_name ? `Stored ${row.image_name}.` : "Scene recorded to memory.",
        tone: "muted"
      };
    case "chat_query":
      return { ...base, ...describeChatQuery(row) };
    case "chat_action_proposed":
      return { ...base, ...describeChatAction(row, "proposed") };
    case "chat_action_executed":
      return { ...base, ...describeChatAction(row, "executed") };
    case "recognize_session":
      return { ...base, ...describeRecognizeSession(row) };
    case "enroll":
      return { ...base, ...describeEnroll(row) };
    case "summary_query":
      return { ...base, ...describeSummaryQuery(row) };
    case "memory_query":
      return { ...base, ...describeMemoryQuery(row) };
    case "object_train":
      return { ...base, ...describeObjectTrain(row) };
    default: {
      const detail = row?.message || row?.detail || row?.event || row?.intent || "";
      return {
        ...base,
        key: "generic",
        title: prettifyEventType(rawType),
        summary: String(detail || "No additional detail recorded.").trim(),
        tone: "info"
      };
    }
  }
}

export { ACTION_LABELS, INTENT_LABELS };
