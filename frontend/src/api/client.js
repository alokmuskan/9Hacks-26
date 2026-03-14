const API_BASE = (import.meta.env.VITE_API_BASE || "http://localhost:8000").replace(/\/+$/, "");
const DEFAULT_TIMEOUT_MS = 15_000;
const LONG_TIMEOUT_MS = 70_000;
const CHAT_TIMEOUT_MS = 35_000;

async function request(path, options = {}) {
  const { timeoutMs = DEFAULT_TIMEOUT_MS, ...fetchOptions } = options;
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
  let resp;
  try {
    resp = await fetch(`${API_BASE}${path}`, {
      ...fetchOptions,
      signal: controller.signal
    });
  } catch (err) {
    clearTimeout(timeoutId);
    if (err?.name === "AbortError") {
      throw new Error(`Request timed out after ${Math.round(timeoutMs / 1000)}s`);
    }
    throw err;
  }
  clearTimeout(timeoutId);

  const contentType = resp.headers.get("content-type") || "";
  const body = contentType.includes("application/json") ? await resp.json() : await resp.text();

  if (!resp.ok) {
    const detail =
      (typeof body === "object" && body && (body.detail || body.message)) ||
      `${resp.status} ${resp.statusText}`;
    throw new Error(String(detail));
  }

  return body;
}

export const api = {
  base: API_BASE,
  status: () => request("/api/v1/monitor/status", { timeoutMs: 12_000 }),
  startMonitor: (payload = {}) =>
    request("/api/v1/monitor/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      timeoutMs: DEFAULT_TIMEOUT_MS
    }),
  stopMonitor: () =>
    request("/api/v1/monitor/stop", {
      method: "POST",
      timeoutMs: DEFAULT_TIMEOUT_MS
    }),
  setToggles: (payload) =>
    request("/api/v1/monitor/toggles", {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      timeoutMs: DEFAULT_TIMEOUT_MS
    }),
  snapshot: () =>
    request("/api/v1/monitor/snapshot", {
      method: "POST",
      timeoutMs: DEFAULT_TIMEOUT_MS
    }),
  detectionsLatest: () => request("/api/v1/detections/latest", { timeoutMs: 12_000 }),
  behaviorLatest: () => request("/api/v1/behavior/latest", { timeoutMs: 12_000 }),
  memoryStats: () => request("/api/v1/memory/stats", { timeoutMs: 12_000 }),
  memoryRecent: (minutes = 5, limit = 20) =>
    request(`/api/v1/memory/recent?minutes=${minutes}&limit=${limit}`, { timeoutMs: 12_000 }),
  memoryFindObject: (name) =>
    request(`/api/v1/memory/find/object?name=${encodeURIComponent(name)}`, { timeoutMs: 12_000 }),
  memoryFindPerson: (name) =>
    request(`/api/v1/memory/find/person?name=${encodeURIComponent(name)}`, { timeoutMs: 12_000 }),
  memorySearch: (text, topK = 5) =>
    request(`/api/v1/memory/search?text=${encodeURIComponent(text)}&top_k=${topK}`, { timeoutMs: 15_000 }),
  logs: ({ eventType, limit = 200 } = {}) => {
    const query = new URLSearchParams();
    query.set("limit", String(limit));
    if (eventType) {
      query.set("event_type", eventType);
    }
    return request(`/api/v1/logs?${query.toString()}`, { timeoutMs: 15_000 });
  },
  sessionSummary: (minutes = 5) => request(`/api/v1/summaries/session?minutes=${minutes}`, { timeoutMs: 20_000 }),
  chatQuery: (payload) =>
    request("/api/v1/chat/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(
        typeof payload === "string"
          ? { message: payload, question: payload }
          : payload || {}
      ),
      timeoutMs: CHAT_TIMEOUT_MS
    }),
  startEnroll: (name, model = "buffalo_sc") =>
    request("/api/v1/enroll/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, model }),
      timeoutMs: DEFAULT_TIMEOUT_MS
    }),
  enrollStatus: () => request("/api/v1/enroll/status", { timeoutMs: 12_000 }),
  stopEnroll: () =>
    request("/api/v1/enroll/stop", {
      method: "POST",
      timeoutMs: DEFAULT_TIMEOUT_MS
    }),
  monitorStartupWaitMs: LONG_TIMEOUT_MS
};

export function wsUrl(path) {
  const base = API_BASE.replace(/^http/, "ws");
  return `${base}${path}`;
}
