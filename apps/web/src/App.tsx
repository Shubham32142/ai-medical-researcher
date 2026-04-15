import { useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { Citation, Message, Trial } from "@curalink/types";

const apiBase = import.meta.env.VITE_API_URL || "/api";

type AssistantExtras = {
  citations?: Citation[];
  trials?: Trial[];
  followUps?: string[];
};

const starterPrompts = [
  "Latest treatment options for stage 3 lung cancer",
  "Ongoing type 1 diabetes trials in the US",
  "Recent evidence for rheumatoid arthritis care",
];

export default function App() {
  const [sessionId, setSessionId] = useState("");
  const [disease, setDisease] = useState("stage 3 lung cancer");
  const [location, setLocation] = useState("Mumbai, India");
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [extrasById, setExtrasById] = useState<Record<string, AssistantExtras>>(
    {},
  );

  const canStart = disease.trim().length > 1 && location.trim().length > 1;

  async function createSession() {
    if (!canStart) return;
    setLoading(true);
    setError("");
    try {
      const res = await fetch(`${apiBase}/sessions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ disease, location }),
      });
      if (!res.ok) throw new Error("Could not create session");
      const data = await res.json();
      setSessionId(data.sessionId);
    } catch {
      setError("Session setup failed. Please ensure the API is running.");
    } finally {
      setLoading(false);
    }
  }

  async function ask(promptText?: string) {
    const content = (promptText ?? question).trim();
    if (!content || !sessionId || loading) return;
    setError("");

    const userMessage: Message = {
      id: crypto.randomUUID(),
      role: "user",
      content,
      createdAt: new Date().toISOString(),
    };

    const assistantId = crypto.randomUUID();
    const assistantMessage: Message = {
      id: assistantId,
      role: "assistant",
      content: "",
      createdAt: new Date().toISOString(),
    };

    setQuestion("");
    setLoading(true);
    setMessages((prev) => [...prev, userMessage, assistantMessage]);
    setExtrasById((prev) => ({
      ...prev,
      [assistantId]: { citations: [], trials: [], followUps: [] },
    }));

    try {
      const response = await fetch(`${apiBase}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sessionId, message: content }),
      });

      if (!response.ok) {
        throw new Error("Chat request failed");
      }

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      if (!reader) {
        setLoading(false);
        return;
      }

      const updateAssistant = (delta: string) => {
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === assistantId
              ? { ...msg, content: `${msg.content}${delta}` }
              : msg,
          ),
        );
      };

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const chunks = buffer.split("\n\n");
        buffer = chunks.pop() || "";

        for (const chunk of chunks) {
          const eventMatch = chunk.match(/event: (.+)/);
          const dataMatch = chunk.match(/data: (.+)/s);
          if (!eventMatch || !dataMatch) continue;
          const event = eventMatch[1].trim();
          const data = JSON.parse(dataMatch[1]);

          if (event === "token") updateAssistant(data.delta);
          if (event === "citations") {
            setExtrasById((prev) => ({
              ...prev,
              [assistantId]: {
                ...prev[assistantId],
                citations: data.citations,
              },
            }));
          }
          if (event === "trials") {
            setExtrasById((prev) => ({
              ...prev,
              [assistantId]: { ...prev[assistantId], trials: data.trials },
            }));
          }
          if (event === "followups") {
            setExtrasById((prev) => ({
              ...prev,
              [assistantId]: {
                ...prev[assistantId],
                followUps: data.followUps,
              },
            }));
          }
        }
      }
    } catch {
      setError("Live research failed. Please retry in a moment.");
    } finally {
      setLoading(false);
    }
  }

  const welcome = useMemo(() => {
    if (!sessionId)
      return "Create a session to start evidence-backed medical research chat.";
    return `Researching ${disease} for ${location}.`;
  }, [sessionId, disease, location]);

  return (
    <div className="page">
      <div className="shell">
        <aside className="sidebar">
          <div>
            <p className="eyebrow">CuraLink</p>
            <h1>AI-powered medical research assistant</h1>
            <p className="muted">
              Grounded in PubMed, OpenAlex, and ClinicalTrials evidence.
            </p>
          </div>

          <div className="card stack">
            <label>
              Disease
              <input
                value={disease}
                onChange={(e) => setDisease(e.target.value)}
                placeholder="Disease or condition"
              />
            </label>
            <label>
              Location
              <input
                value={location}
                onChange={(e) => setLocation(e.target.value)}
                placeholder="City, country"
              />
            </label>
            <button
              className="primary"
              onClick={createSession}
              disabled={!canStart || loading}
            >
              {sessionId
                ? "Session ready"
                : loading
                  ? "Starting..."
                  : "Start session"}
            </button>
          </div>

          <div className="card">
            <p className="muted small">Quick prompts</p>
            <div className="chips">
              {starterPrompts.map((prompt) => (
                <button
                  key={prompt}
                  className="chip"
                  onClick={() => ask(prompt)}
                  disabled={!sessionId || loading}
                >
                  {prompt}
                </button>
              ))}
            </div>
          </div>
        </aside>

        <main className="main">
          <div className="hero card">
            <p className="status">{welcome}</p>
            <p className="small muted">
              Educational use only. This prototype summarizes research evidence
              and does not replace clinical advice.
            </p>
            {error ? <div className="alert">{error}</div> : null}
          </div>

          <div className="messages">
            {messages.length === 0 ? (
              <div className="card empty">
                <h2>Ask a question</h2>
                <p className="muted">
                  Examples: treatment updates, trial availability, supplement
                  interactions, or clinical evidence.
                </p>
              </div>
            ) : (
              messages.map((message) => {
                const extras = extrasById[message.id];
                return (
                  <section
                    key={message.id}
                    className={`bubble ${message.role}`}
                  >
                    <div className="bubble-head">
                      {message.role === "user" ? "You" : "CuraLink"}
                    </div>
                    {message.role === "assistant" ? (
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>
                        {message.content || "Thinking..."}
                      </ReactMarkdown>
                    ) : (
                      <p>{message.content}</p>
                    )}

                    {extras?.citations?.length ? (
                      <div className="panel">
                        <h3>Citations</h3>
                        {extras.citations.map((citation) => (
                          <a
                            key={citation.id}
                            href={citation.url}
                            target="_blank"
                            rel="noreferrer"
                            className="resource"
                          >
                            <strong>{citation.title}</strong>
                            <span>
                              {citation.source.toUpperCase()} • {citation.year}
                            </span>
                          </a>
                        ))}
                      </div>
                    ) : null}

                    {extras?.trials?.length ? (
                      <div className="panel">
                        <h3>Trials</h3>
                        {extras.trials.map((trial) => (
                          <a
                            key={trial.nctId}
                            href={trial.url}
                            target="_blank"
                            rel="noreferrer"
                            className="resource"
                          >
                            <strong>{trial.title}</strong>
                            <span>
                              {trial.nctId} • {trial.status}
                            </span>
                          </a>
                        ))}
                      </div>
                    ) : null}

                    {extras?.followUps?.length ? (
                      <div className="chips">
                        {extras.followUps.map((item) => (
                          <button
                            key={item}
                            className="chip"
                            onClick={() => ask(item)}
                            disabled={loading}
                          >
                            {item}
                          </button>
                        ))}
                      </div>
                    ) : null}
                  </section>
                );
              })
            )}
          </div>

          <div className="composer card">
            <textarea
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Ask about treatment options, recent evidence, side effects, or active trials..."
              rows={4}
            />
            <button
              className="primary"
              onClick={() => ask()}
              disabled={!sessionId || loading || !question.trim()}
            >
              {loading ? "Researching..." : "Send"}
            </button>
          </div>
        </main>
      </div>
    </div>
  );
}
