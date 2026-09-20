import { describe, expect, it } from "vitest";
import { describeEvent, humanizeSeconds, prettifyEventType, shortSessionId } from "./eventDisplay";
import { buildSessionInsights } from "./sessionInsights";

// Fixtures are trimmed copies of real metrics_log.jsonl rows, so the tests fail
// if the backend's emitted shape drifts.
const recognizeSession = (overrides = {}) => ({
  timestamp_utc: "2026-09-20T05:46:02.000000+00:00",
  event_type: "recognize_session",
  session_id: "monitor-20260920-104542",
  duration_sec: 15.225,
  aggregate: {
    session_id: "monitor-20260920-104542",
    frames_total: 38,
    detections_total: 38,
    known_detections: 38,
    unknown_detections: 2,
    unique_individuals_seen: 1,
    unknown_alert_events: 1,
    unknown_alert_density_per_min: 3.94,
    object_detections_total: 41,
    object_class_counts_total: { person: 38, laptop: 3 },
    active_subjects: [{ name: "alok", confidence: 0.8141 }],
    active_objects: ["person"],
    behavior_attention_map: { alok: { laptop: 53.378 } },
    peak_simultaneous_faces: 2,
    avg_fps: 0.68,
    memory_snapshots_auto: 7,
    ...(overrides.aggregate || {})
  },
  ...overrides
});

describe("humanizeSeconds", () => {
  it("formats sub-minute and multi-minute durations", () => {
    expect(humanizeSeconds(0)).toBe("0s");
    expect(humanizeSeconds(53.378)).toBe("53s");
    expect(humanizeSeconds(128)).toBe("2m 8s");
    expect(humanizeSeconds(120)).toBe("2m");
  });
});

describe("prettifyEventType / shortSessionId", () => {
  it("prettifies unknown snake_case types instead of leaking them", () => {
    expect(prettifyEventType("some_new_event")).toBe("Some new event");
    expect(prettifyEventType("")).toBe("Event");
  });

  it("keeps only the readable part of a session id", () => {
    expect(shortSessionId("monitor-20260920-104542")).toBe("20260920-104542");
    expect(shortSessionId("plain")).toBe("plain");
  });
});

describe("describeEvent", () => {
  it("turns a behaviour event into a sentence", () => {
    const start = describeEvent({
      event_type: "behavior_event",
      event: "start",
      person: "alok",
      target_object: "person",
      duration_sec: 0
    });
    expect(start.title).toBe("alok entered view");
    expect(start.tone).toBe("info");

    const end = describeEvent({
      event_type: "behavior_event",
      event: "end",
      person: "alok",
      target_object: "laptop",
      duration_sec: 53.378
    });
    expect(end.title).toBe("alok — focus ended");
    expect(end.summary).toContain("53s");
  });

  it("explains where a chat answer came from", () => {
    const fromData = describeEvent({
      event_type: "chat_query",
      question: "How many people?",
      intent: "person_count",
      hit: true,
      used_llm: false
    });
    expect(fromData.title).toBe("Question asked");
    expect(fromData.summary).toContain("How many people?");
    expect(fromData.summary).toContain("people count");
    expect(fromData.summary).toContain("recorded data");

    const viaLlm = describeEvent({
      event_type: "chat_query",
      question: "What happened?",
      intent: "open_ended",
      hit: true,
      used_llm: true
    });
    expect(viaLlm.summary).toContain("LLM");

    const miss = describeEvent({ event_type: "chat_query", question: "?", intent: "presence", hit: false });
    expect(miss.tone).toBe("warn");
  });

  it("labels proposed vs executed actions", () => {
    expect(describeEvent({ event_type: "chat_action_proposed", action: "monitor_stop" }).summary).toContain(
      "Stop monitoring"
    );
    expect(describeEvent({ event_type: "chat_action_executed", action: "monitor_stop" }).tone).toBe("success");
  });

  it("summarises sessions, enrollments and summaries", () => {
    const session = describeEvent(recognizeSession());
    expect(session.summary).toContain("38 detections");
    expect(session.summary).toContain("1 known person");
    expect(session.summary).toContain("1 unknown alert");
    expect(session.tone).toBe("warn");

    const enroll = describeEvent({
      event_type: "enroll",
      name: "alok",
      samples_captured: 25,
      duration_sec: 18.952
    });
    expect(enroll.summary).toBe("alok · 25 samples captured in 19s.");

    expect(describeEvent({ event_type: "summary_query", minutes: 5, hit: true }).summary).toContain("last 5 minutes");
  });

  it("degrades unknown event types gracefully", () => {
    const row = describeEvent({ event_type: "brand_new_thing", message: "something happened" });
    expect(row.title).toBe("Brand new thing");
    expect(row.summary).toBe("something happened");
    expect(row.rawType).toBe("brand_new_thing");
  });
});

describe("buildSessionInsights", () => {
  it("prefers live counts and mirrors the card sentinel fallback", () => {
    const insights = buildSessionInsights({
      logs: [recognizeSession()],
      detections: {
        counts: {
          face_count: 2,
          object_count: 3,
          known_detections: 45,
          unknown_detections: 8,
          known_unique: 4,
          unknown_alerts: 2
        },
        active_subjects: [{ name: "alok", confidence: 0.9 }],
        active_objects: ["person"]
      }
    });

    expect(insights.totals).toEqual({ known: 4, unknown: 2, total: 6, source: "live" });
    expect(insights.live.faces).toBe(2);
    expect(insights.live.peopleTracked).toBe(4);
    expect(insights.live.activeSubjects[0].name).toBe("alok");
  });

  it("falls back to cumulative counters when the unique fields are absent", () => {
    const insights = buildSessionInsights({
      detections: { counts: { known_detections: 12, unknown_detections: 3 } }
    });
    expect(insights.totals).toEqual({ known: 12, unknown: 3, total: 15, source: "live" });
  });

  it("falls back to the newest recorded session when idle", () => {
    const insights = buildSessionInsights({ logs: [recognizeSession()] });
    expect(insights.totals.source).toBe("session");
    expect(insights.totals.known).toBe(1);
    expect(insights.totals.unknown).toBe(1);
    expect(insights.totals.total).toBe(2);
  });

  it("lists sessions newest-first with the drill-down fields", () => {
    const older = recognizeSession({
      timestamp_utc: "2026-09-20T04:00:00+00:00",
      session_id: "monitor-20260920-090000",
      aggregate: { session_id: "monitor-20260920-090000", detections_total: 10 }
    });
    const insights = buildSessionInsights({ logs: [older, recognizeSession()] });

    expect(insights.sessions).toHaveLength(2);
    expect(insights.sessions[0].timestampUtc).toBe("2026-09-20T05:46:02.000000+00:00");
    expect(insights.sessions[0].detections).toBe(38);
    expect(insights.sessions[0].unknownAlerts).toBe(1);
    expect(insights.sessions[0].durationSec).toBe(15.225);
    expect(insights.detectionTotals.detections).toBe(48);
  });

  it("aggregates known people across sessions, behaviour and enrollment", () => {
    const insights = buildSessionInsights({
      logs: [
        { event_type: "enroll", name: "alok", timestamp_utc: "2026-09-19T14:07:16+00:00", samples_captured: 25 },
        recognizeSession(),
        { event_type: "behavior_event", person: "alok", event: "start", timestamp_utc: "2026-09-20T05:50:00+00:00" }
      ]
    });

    expect(insights.knownPeople).toHaveLength(1);
    const alok = insights.knownPeople[0];
    expect(alok.name).toBe("alok");
    expect(alok.enrolled).toBe(true);
    expect(alok.bestConfidence).toBe(0.814);
    expect(alok.sessionsSeen).toBe(1);
    expect(alok.attentionSec).toBe(53.4);
    expect(alok.lastSeenUtc).toBe("2026-09-20T05:50:00+00:00");
    expect(alok.currentlyVisible).toBe(false);
  });

  it("marks the person visible in the live payload", () => {
    const insights = buildSessionInsights({
      logs: [recognizeSession()],
      detections: { counts: { known_detections: 1, unknown_detections: 0 }, active_subjects: [{ name: "alok", confidence: 0.7 }] }
    });
    expect(insights.knownPeople[0].currentlyVisible).toBe(true);
  });

  it("totals unknown alerts with a per-minute rate over monitored time", () => {
    const insights = buildSessionInsights({
      logs: [
        recognizeSession({ duration_sec: 30, aggregate: { unknown_alert_events: 3, unknown_detections: 5 } }),
        recognizeSession({ timestamp_utc: "2026-09-20T04:00:00+00:00", duration_sec: 30, aggregate: { unknown_alert_events: 1 } })
      ]
    });

    expect(insights.unknownAlerts.total).toBe(4);
    expect(insights.unknownAlerts.unknownFrames).toBe(5);
    expect(insights.unknownAlerts.sessionsWithAlerts).toBe(2);
    // 4 alerts over 60s of monitored time.
    expect(insights.unknownAlerts.densityPerMin).toBe(4);
  });

  it("sums object classes across sessions", () => {
    const insights = buildSessionInsights({ logs: [recognizeSession(), recognizeSession()] });
    expect(insights.objectClasses).toEqual([
      { label: "person", count: 76 },
      { label: "laptop", count: 6 }
    ]);
  });

  it("returns an empty, harmless shape with no data", () => {
    const insights = buildSessionInsights({});
    expect(insights.totals).toEqual({ known: 0, unknown: 0, total: 0, source: "none" });
    expect(insights.sessions).toEqual([]);
    expect(insights.knownPeople).toEqual([]);
    expect(insights.unknownAlerts.total).toBe(0);
    expect(insights.unknownAlerts.densityPerMin).toBe(0);
    expect(insights.objectClasses).toEqual([]);
  });
});
