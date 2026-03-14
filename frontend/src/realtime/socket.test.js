import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createEventSocket } from "./socket";

vi.mock("../api/client", () => ({
  wsUrl: vi.fn(() => "ws://test.local/api/v1/stream/events/ws")
}));

class MockWebSocket {
  static instances = [];

  constructor(url) {
    this.url = url;
    this.readyState = 0;
    this.onopen = null;
    this.onmessage = null;
    this.onerror = null;
    this.onclose = null;
    MockWebSocket.instances.push(this);
  }

  emitOpen() {
    this.readyState = 1;
    this.onopen?.({});
  }

  emitMessage(payload) {
    this.onmessage?.({ data: JSON.stringify(payload) });
  }

  emitClose() {
    this.readyState = 3;
    this.onclose?.({});
  }

  close() {
    this.emitClose();
  }
}

describe("createEventSocket", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    MockWebSocket.instances = [];
    globalThis.WebSocket = MockWebSocket;
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
    delete globalThis.WebSocket;
  });

  it("forwards parsed websocket events", () => {
    const events = [];
    const states = [];
    const socket = createEventSocket({
      onEvent: (evt) => events.push(evt),
      onState: (state) => states.push(state)
    });

    const ws = MockWebSocket.instances[0];
    ws.emitOpen();
    ws.emitMessage({ type: "detections", payload: { sequence: 42 } });

    expect(states.some((s) => s.connected)).toBe(true);
    expect(events).toHaveLength(1);
    expect(events[0].type).toBe("detections");
    socket.close();
  });

  it("reconnects with backoff after unexpected close", () => {
    const states = [];
    const socket = createEventSocket({
      onEvent: () => {},
      onState: (state) => states.push(state)
    });

    const first = MockWebSocket.instances[0];
    first.emitOpen();
    first.emitClose();

    expect(states.at(-1)?.reconnecting).toBe(true);
    vi.advanceTimersByTime(1000);
    expect(MockWebSocket.instances).toHaveLength(2);

    socket.close();
    const countAfterClose = MockWebSocket.instances.length;
    vi.advanceTimersByTime(15_000);
    expect(MockWebSocket.instances).toHaveLength(countAfterClose);
  });
});
