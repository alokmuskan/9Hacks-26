// Card drill-down data for the dashboard.
//
// The dashboard cards are thin views over data the app already holds — the
// metrics log (`logs`), the live per-frame payload (`detections`), and the
// behaviour tracker (`behavior`). Nothing here talks to the network: this module
// only reshapes those three inputs into the lists the cards open onto, so the
// modal stays instant and works while the pipeline is idle.
//
// Pure and dependency-free for unit testing. Shapes verified against real
// metrics_log.jsonl rows and the /api/v1/detections/latest payload.

function num(value, fallback = 0) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function isPlainObject(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function normalizeSubjects(subjects) {
  if (!Array.isArray(subjects)) {
    return [];
  }
  return subjects
    .filter((row) => row && typeof row === "object")
    .map((row) => ({
      name: String(row.name || row.label || "unknown").trim() || "unknown",
      confidence: num(row.confidence ?? row.score, 0)
    }));
}

function sumValues(map) {
  if (!isPlainObject(map)) {
    return 0;
  }
  return Object.values(map).reduce((total, value) => total + num(value, 0), 0);
}

function mapSession(row) {
  const aggregate = isPlainObject(row?.aggregate) ? row.aggregate : {};
  const detections = num(
    aggregate.detections_total,
    num(aggregate.known_detections) + num(aggregate.unknown_detections)
  );
  return {
    id: String(aggregate.session_id || row?.session_id || "").trim(),
    timestampUtc: row?.timestamp_utc || null,
    startUtc: row?.start_utc || null,
    endUtc: row?.end_utc || null,
    durationSec: num(row?.duration_sec, 0),
    model: String(row?.model || "").trim(),
    frames: num(aggregate.frames_total, 0),
    framesWithFaces: num(aggregate.frames_with_faces, 0),
    detections,
    knownFrames: num(aggregate.known_detections, 0),
    unknownFrames: num(aggregate.unknown_detections, 0),
    knownPeople: num(aggregate.unique_individuals_seen, 0),
    unknownAlerts: num(aggregate.unknown_alert_events, 0),
    unknownDensityPerMin: num(aggregate.unknown_alert_density_per_min, 0),
    objects: num(aggregate.object_detections_total, 0),
    peakFaces: num(aggregate.peak_simultaneous_faces, 0),
    avgFps: num(aggregate.avg_fps, 0),
    recognitionRate: num(aggregate.recognition_rate, 0),
    snapshotsAuto: num(aggregate.memory_snapshots_auto, 0),
    subjects: normalizeSubjects(aggregate.active_subjects),
    objectsSeen: Array.isArray(aggregate.active_objects) ? aggregate.active_objects.map(String) : [],
    objectClasses: isPlainObject(aggregate.object_class_counts_total) ? aggregate.object_class_counts_total : {},
    attentionMap: isPlainObject(aggregate.behavior_attention_map) ? aggregate.behavior_attention_map : {},
    attentionTotalSec: num(aggregate.behavior_attention_total_sec, 0),
    behaviorEvents: num(aggregate.behavior_events_count, 0),
    chatQueries: num(aggregate.chat_queries_total, 0)
  };
}

/**
 * Reshape the dashboard's data sources into drill-down lists.
 *
 * @param {{logs?: Array<object>, detections?: object|null, behavior?: object|null}} input
 */
export function buildSessionInsights({ logs = [], detections = null, behavior = null } = {}) {
  const rows = Array.isArray(logs) ? logs : [];
  const sessions = [];
  const enrollments = [];
  const behaviorEvents = [];

  for (const row of rows) {
    const type = String(row?.event_type || row?.type || "");
    if (type === "recognize_session") {
      sessions.push(mapSession(row));
    } else if (type === "enroll") {
      enrollments.push({
        name: String(row?.name || "").trim(),
        timestampUtc: row?.timestamp_utc || null,
        samples: num(row?.samples_captured, num(row?.total_samples_for_name, 0))
      });
    } else if (type === "behavior_event") {
      behaviorEvents.push(row);
    }
  }

  // The metrics log is appended oldest-first; the UI lists newest-first.
  sessions.reverse();

  const latestSession = sessions[0] || null;
  const counts = isPlainObject(detections?.counts) ? detections.counts : null;
  const hasLiveCounts =
    Boolean(counts) && ("known_detections" in counts || "unknown_detections" in counts);

  // Sentinel handling mirrors the card metrics: -1 means "field absent", so fall
  // back to the cumulative per-frame counters for older backends.
  const liveKnownUnique = num(counts?.known_unique, -1);
  const liveKnown = liveKnownUnique >= 0 ? liveKnownUnique : num(counts?.known_detections, 0);
  const liveAlerts = num(counts?.unknown_alerts, -1);
  const liveUnknown = liveAlerts >= 0 ? liveAlerts : num(counts?.unknown_detections, 0);

  const live = {
    available: hasLiveCounts,
    sequence: num(detections?.sequence, 0),
    timestampUtc: detections?.timestamp_utc || null,
    // Instantaneous frame contents.
    faces: num(counts?.face_count, 0),
    objects: num(counts?.object_count, 0),
    // Cumulative for the running session.
    peopleTracked: hasLiveCounts ? liveKnown : 0,
    unknownAlerts: hasLiveCounts ? liveUnknown : 0,
    knownFrames: num(counts?.known_detections, 0),
    unknownFrames: num(counts?.unknown_detections, 0),
    activeSubjects: normalizeSubjects(detections?.active_subjects),
    activeObjects: Array.isArray(detections?.active_objects) ? detections.active_objects.map(String) : [],
    attention: Array.isArray(detections?.attention) ? detections.attention : []
  };

  const totals = hasLiveCounts
    ? { known: liveKnown, unknown: liveUnknown, total: liveKnown + liveUnknown, source: "live" }
    : {
        known: latestSession ? latestSession.knownPeople : 0,
        unknown: latestSession ? latestSession.unknownAlerts : 0,
        total: latestSession ? latestSession.knownPeople + latestSession.unknownAlerts : 0,
        source: latestSession ? "session" : "none"
      };

  // ── Known people ───────────────────────────────────────────────────────────
  const people = new Map();
  const touch = (rawName) => {
    const name = String(rawName || "").trim();
    if (!name) {
      return null;
    }
    if (!people.has(name)) {
      people.set(name, {
        name,
        bestConfidence: 0,
        sessionsSeen: 0,
        attentionSec: 0,
        lastSeenUtc: null,
        firstSeenUtc: null,
        enrolled: false,
        currentlyVisible: false
      });
    }
    return people.get(name);
  };

  for (const session of sessions) {
    for (const subject of session.subjects) {
      const person = touch(subject.name);
      if (!person) {
        continue;
      }
      person.sessionsSeen += 1;
      person.bestConfidence = Math.max(person.bestConfidence, subject.confidence);
      person.lastSeenUtc = person.lastSeenUtc || session.timestampUtc;
      person.firstSeenUtc = session.timestampUtc || person.firstSeenUtc;
    }
    for (const [name, byObject] of Object.entries(session.attentionMap)) {
      const person = touch(name);
      if (person) {
        person.attentionSec += sumValues(byObject);
      }
    }
  }

  for (const event of behaviorEvents) {
    const person = touch(event?.person);
    if (!person) {
      continue;
    }
    const stamp = event?.timestamp_utc || event?.event_time_utc || null;
    if (stamp && (!person.lastSeenUtc || Date.parse(stamp) > Date.parse(person.lastSeenUtc))) {
      person.lastSeenUtc = stamp;
    }
  }

  for (const enrollment of enrollments) {
    const person = touch(enrollment.name);
    if (person) {
      person.enrolled = true;
      person.firstSeenUtc = person.firstSeenUtc || enrollment.timestampUtc;
    }
  }

  for (const subject of live.activeSubjects) {
    const person = touch(subject.name);
    if (person) {
      person.currentlyVisible = true;
      person.bestConfidence = Math.max(person.bestConfidence, subject.confidence);
    }
  }

  for (const person of people.values()) {
    person.attentionSec = Math.round(person.attentionSec * 10) / 10;
    person.bestConfidence = Math.round(person.bestConfidence * 1000) / 1000;
  }

  const knownPeople = [...people.values()].sort((a, b) => {
    if (a.currentlyVisible !== b.currentlyVisible) {
      return a.currentlyVisible ? -1 : 1;
    }
    const aStamp = Date.parse(a.lastSeenUtc || "") || 0;
    const bStamp = Date.parse(b.lastSeenUtc || "") || 0;
    if (aStamp !== bStamp) {
      return bStamp - aStamp;
    }
    return b.attentionSec - a.attentionSec;
  });

  // ── Unknown alerts ─────────────────────────────────────────────────────────
  const alertSessions = sessions.filter((session) => session.unknownAlerts > 0);
  const totalAlerts = sessions.reduce((total, session) => total + session.unknownAlerts, 0);
  const totalMinutes = sessions.reduce((total, session) => total + session.durationSec, 0) / 60;
  const unknownAlerts = {
    total: totalAlerts,
    liveCount: hasLiveCounts ? liveUnknown : 0,
    unknownFrames: sessions.reduce((total, session) => total + session.unknownFrames, 0),
    sessionsWithAlerts: alertSessions.length,
    totalSessions: sessions.length,
    densityPerMin: totalMinutes > 0 ? Math.round((totalAlerts / totalMinutes) * 100) / 100 : 0,
    latestUtc: alertSessions[0]?.timestampUtc || null,
    sessions: alertSessions.slice(0, 12)
  };

  // ── Object classes ─────────────────────────────────────────────────────────
  const classTotals = new Map();
  for (const session of sessions) {
    for (const [label, count] of Object.entries(session.objectClasses)) {
      classTotals.set(label, (classTotals.get(label) || 0) + num(count, 0));
    }
  }
  const objectClasses = [...classTotals.entries()]
    .map(([label, count]) => ({ label, count }))
    .sort((a, b) => b.count - a.count);

  const detectionTotals = {
    sessions: sessions.length,
    detections: sessions.reduce((total, session) => total + session.detections, 0),
    objects: sessions.reduce((total, session) => total + session.objects, 0),
    frames: sessions.reduce((total, session) => total + session.frames, 0),
    monitoredSec: sessions.reduce((total, session) => total + session.durationSec, 0)
  };

  return {
    live,
    totals,
    sessions: sessions.slice(0, 12),
    sessionCount: sessions.length,
    detectionTotals,
    objectClasses: objectClasses.slice(0, 12),
    knownPeople,
    unknownAlerts,
    enrollments,
    behavior,
    lastUpdatedUtc: latestSession?.timestampUtc || live.timestampUtc || null
  };
}
