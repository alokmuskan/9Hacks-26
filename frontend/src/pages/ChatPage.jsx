import { useContext, useState } from "react";
import { Send } from "lucide-react";
import { SurveillanceContext } from "../context/SurveillanceContext";
import "./chat.css";

function renderAssistant(row) {
  if (row.reply) {
    return row.reply;
  }
  if (row.answer) {
    return row.answer;
  }
  if (row.intent) {
    return `Intent ${row.intent} handled.`;
  }
  return "No response payload.";
}

function ChatPage() {
  const { chatHistory, askChat, confirmChatAction, dismissChatProposal } = useContext(SurveillanceContext);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);

  const submit = async (e) => {
    e.preventDefault();
    const q = input.trim();
    if (!q || busy) {
      return;
    }
    try {
      setBusy(true);
      setInput("");
      await askChat(q);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="page-content">
      <div className="page-header">
        <h1 className="page-title">Chatbot</h1>
        <p className="page-subtitle">Deterministic memory tools + Groq fallback through backend API.</p>
      </div>

      <section className="glass-panel chat-card">
        <div className="chat-scroll">
          {!chatHistory.length && <div className="empty-cell">Ask about recent activity, last-seen, or summaries.</div>}
          {chatHistory.map((row, idx) => (
            <div key={`${row.ts || idx}-${idx}`} className={`chat-msg ${row.role}`}>
              <div className="tiny">{row.role === "user" ? "You" : "Assistant"}</div>
              <div>{row.role === "user" ? row.question : renderAssistant(row)}</div>
              {row.role === "assistant" && Array.isArray(row.citations) && row.citations.length > 0 && (
                <div className="chat-citations">
                  {row.citations.map((citation) => (
                    <div key={citation.id} className="chat-citation-item">
                      <strong>{citation.id}</strong>
                      <span>{citation.source}</span>
                      <span>{citation.timestamp || "-"}</span>
                      <span>{citation.detail}</span>
                    </div>
                  ))}
                </div>
              )}
              {row.role === "assistant" && row.proposed_action?.confirm_action_id && (
                <div className="chat-action-row">
                  <button
                    className="btn-primary"
                    disabled={busy}
                    onClick={async () => {
                      try {
                        setBusy(true);
                        await confirmChatAction(row.proposed_action.confirm_action_id);
                      } finally {
                        setBusy(false);
                      }
                    }}
                  >
                    Confirm Action
                  </button>
                  <button
                    className="btn-secondary"
                    disabled={busy}
                    onClick={() => dismissChatProposal(row.proposed_action.confirm_action_id)}
                  >
                    Cancel
                  </button>
                </div>
              )}
            </div>
          ))}
        </div>
        <form className="chat-form" onSubmit={submit}>
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="What happened in the last 5 minutes?"
          />
          <button type="submit" className="btn-primary" disabled={busy || !input.trim()}>
            <Send size={16} />
            Send
          </button>
        </form>
      </section>
    </div>
  );
}

export default ChatPage;
