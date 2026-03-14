import { wsUrl } from "../api/client";

export function createEventSocket({ onEvent, onState }) {
  let ws = null;
  let closedByClient = false;
  let retryMs = 1000;
  let retryTimer = null;

  const connect = () => {
    onState?.({ connected: false, reconnecting: retryMs > 1000 });
    ws = new WebSocket(wsUrl("/api/v1/stream/events/ws"));

    ws.onopen = () => {
      retryMs = 1000;
      onState?.({ connected: true, reconnecting: false });
    };

    ws.onmessage = (evt) => {
      try {
        const payload = JSON.parse(evt.data);
        onEvent?.(payload);
      } catch (_err) {
        // Ignore malformed events.
      }
    };

    ws.onerror = () => {
      onState?.({ connected: false, reconnecting: true });
    };

    ws.onclose = () => {
      onState?.({ connected: false, reconnecting: !closedByClient });
      if (!closedByClient) {
        retryTimer = setTimeout(connect, retryMs);
        retryMs = Math.min(retryMs * 1.6, 10_000);
      }
    };
  };

  connect();

  return {
    close: () => {
      closedByClient = true;
      if (retryTimer) {
        clearTimeout(retryTimer);
      }
      if (ws) {
        ws.close();
      }
    }
  };
}
