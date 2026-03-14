# Chat and Situation Summary

## Chat Command

### Interactive REPL

```bash
pixi run python main.py chat
```

Type queries, `quit` to exit.

### One-shot

```bash
pixi run python main.py chat --question "When did you last see a laptop?"
```

## Runtime Chat

While `recognize` is running, press `c` and enter a query.

Runtime mode can execute actions (example: snapshot capture) because it has access to the live frame buffer.

## Supported Deterministic Intents

- Situation summary:
  - “What happened in the last 5 minutes?”
  - “Recent activity”
- Memory stats
- Recent snapshots
- Last seen object
- Last seen person
- Presence:
  - “Who is present?”
- Attention:
  - “What is Hemanth looking at?”
- Snapshot action:
  - “Take snapshot” / “Capture snapshot”

If no deterministic intent matches, Groq fallback is attempted (if API key is configured).

## Situation Summary Command

```bash
pixi run python main.py session-summary --minutes 5
pixi run python main.py session-summary --minutes 10 --json
```

Output is narrative-first and deterministic, for example:

- top person-object attention lines
- snapshot count line
- most viewed object line

If no qualifying activity exists:

`No notable activity was recorded in the last N minutes.`

## Groq Configuration

Set one of:

- `GROQ_API_KEY` (preferred)
- `groq_api_key` (fallback)

Optional:

- `GROQ_MODEL` (default: `llama-3.3-70b-versatile`)

## Telemetry

Each chat request appends a `chat_query` event:

- question
- intent
- action
- hit/miss
- used_llm
- duration

Each summary request appends a `summary_query` event.
