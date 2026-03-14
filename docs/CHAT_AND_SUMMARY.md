# Chat and Situation Summary

## CLI Chat (`main.py`)

### Interactive REPL

```bash
pixi run python main.py chat
```

Type queries and use `quit` to exit.

### One-shot

```bash
pixi run python main.py chat --question "When did you last see a laptop?"
```

## Runtime Chat During `recognize`

While `recognize` is running, press `c` and enter a query.

Runtime mode can perform live-only actions (for example, manual snapshot capture) because it has the current frame buffer in memory.

## API Chat (`/api/v1/chat/query`)

Request fields:

- `message` or `question`
- `session_id` (optional, created if omitted)
- `confirm_action_id` (optional, for action execution)

Typical request:

```json
{
  "message": "What happened in the last 10 minutes?"
}
```

Response fields:

- `session_id`
- `reply` and `answer`
- `intent`, `action`, `hit`, `used_llm`, `grounded`
- `citations`
- optional `proposed_action`, `executed_action`, `summary`, `snapshot`

## Deterministic Intents

Deterministic handling is attempted before LLM fallback:

- situation summary (`what happened in the last N minutes`, `recent activity`)
- memory stats
- recent snapshots
- last seen object
- last seen person
- presence (`who is present`)
- attention (`what is <person> looking at`)

If deterministic handling cannot answer, API chat builds grounding from metrics/memory/runtime context and only then attempts Groq fallback.

## Action Confirmation Flow (API Chat)

Mutating actions are two-step in API mode.

Step 1: ask for an action.

```json
{
  "message": "Turn off general yolo"
}
```

The response includes `proposed_action.confirm_action_id`.

Step 2: confirm execution.

```json
{
  "session_id": "chat-20260314-113500-a1b2c3d4",
  "confirm_action_id": "7f3d1f..."
}
```

Supported confirmed actions:

- capture snapshot
- start monitoring
- stop monitoring
- run summary
- toggle general YOLO
- toggle custom YOLO
- toggle gaze

Invalid or expired confirmation IDs return a safe failure (`intent: action_confirm`, `hit: false`).

## Situation Summary Endpoints

CLI:

```bash
pixi run python main.py session-summary --minutes 5
pixi run python main.py session-summary --minutes 10 --json
```

API:

`GET /api/v1/summaries/session?minutes=5&json=false`

Response:

- `summary` (rendered text unless `json=true`)
- `rendered` (always text)
- `json` (structured summary object)

If no qualifying activity exists, the renderer returns:

`No notable activity was recorded in the last N minutes.`

## Citations and Grounding (API Chat)

Citations are compact records attached to chat responses:

- `id` (for example `C1`)
- `source`
- `timestamp`
- `detail`

They are sourced from recent metrics rows, latest session aggregate, memory hits, and recent snapshots.

## Groq Configuration

Set one of:

- `GROQ_API_KEY` (preferred)
- `groq_api_key` (fallback)

Optional:

- `GROQ_MODEL` (default: `llama-3.3-70b-versatile`)

## Telemetry

Each chat request appends `chat_query` with:

- question
- intent
- action
- hit
- used_llm
- grounded
- duration
- session id (API mode)

Summary calls append `summary_query`.

API action flow also appends:

- `chat_action_proposed`
- `chat_action_executed`
