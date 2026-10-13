import { useState, useRef, useEffect } from "react";

interface Message {
  role: "user" | "assistant";
  content: string;
}

interface ChatConfig {
  enabled: boolean;
  reason?: string | null;
  model?: string;
  reasoning_effort?: string;
  base_url?: string;
  remaining_calls?: number;
  controls?: { max_output_tokens: number };
}

export default function ChatPanel() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [config, setConfig] = useState<ChatConfig | null>(null);
  const [consent, setConsent] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const refreshConfig = async () => {
    try {
      const response = await fetch("/api/chat-config");
      if (!response.ok) throw new Error("Configuration unavailable");
      setConfig(await response.json());
    } catch {
      setConfig({ enabled: false, reason: "Research chat configuration unavailable." });
    }
  };

  useEffect(() => { void refreshConfig(); }, []);

  const sendMessage = async () => {
    if (!input.trim() || loading || !config?.enabled || !consent) return;

    const userMessage = input.trim();
    setInput("");
    setConsent(false);
    setLoading(true);

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: userMessage, consent_to_send: true }),
      });

      const data = await response.json();
      if (data.error) {
        setMessages((prev) => [
          ...prev,
          { role: "assistant", content: `Error: ${data.error}` },
        ]);
      } else {
        setMessages((prev) => [
          ...prev,
          { role: "user", content: userMessage },
          { role: "assistant", content: data.answer || "No response" },
        ]);
      }
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: `Failed to get response: ${err}` },
      ]);
    } finally {
      setLoading(false);
      await refreshConfig();
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="chat-panel">
      <div className="chat-header">
        <span>Public research questions</span>
      </div>
      <div className="chat-settings" role="status">
        {config?.enabled ? (
          <>
            <div>{config.model} · {config.reasoning_effort} reasoning · {config.remaining_calls} calls remaining</div>
            <div>Output cap: {config.controls?.max_output_tokens} tokens. Each question is independent.</div>
          </>
        ) : <div>{config?.reason || "Checking research chat configuration..."}</div>}
        <div>Experiment data and results stay local. Use the status and board panels for run updates.</div>
      </div>
      <div className="chat-messages">
        {messages.length === 0 && (
          <div className="chat-empty">
            Ask about general research methods:
            <ul>
              <li>"How does beta-neutral weighting work?"</li>
              <li>"Why lag accounting information?"</li>
            </ul>
          </div>
        )}
        {messages.map((msg, i) => (
          <div key={i} className={`chat-message ${msg.role}`}>
            <div className="chat-message-content">{msg.content}</div>
          </div>
        ))}
        {loading && (
          <div className="chat-message assistant">
            <div className="chat-message-content chat-loading">Thinking...</div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>
      {config?.enabled && (
        <label className="chat-consent">
          <input type="checkbox" checked={consent} disabled={loading || !input.trim()}
            onChange={(e) => setConsent(e.target.checked)} />
          Send this public question to {config.base_url}
        </label>
      )}
      <div className="chat-input-area">
        <input
          type="text"
          value={input}
          onChange={(e) => { setInput(e.target.value); setConsent(false); }}
          onKeyDown={handleKeyDown}
          placeholder="Ask a public research question..."
          aria-label="Public research question"
          disabled={loading || !config?.enabled}
        />
        <button onClick={sendMessage} disabled={loading || !input.trim() || !config?.enabled || !consent}>
          Send
        </button>
      </div>
    </div>
  );
}
