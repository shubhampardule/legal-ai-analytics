import { useEffect, useMemo, useState } from "react";
import { MessageSquare, Send, Loader2, Info } from "lucide-react";
import { chatWithCase } from "../api/client";

function buildSourceKey({ inputMode, caseIdValue, textValue }) {
  if (inputMode === "caseId") return `case:${caseIdValue || ""}`;
  return `text:${(textValue || "").slice(0, 120)}`;
}

export function CaseChatCard({ inputMode, caseIdValue, textValue, disabled }) {
  const [messages, setMessages] = useState([]);
  const [question, setQuestion] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const sourceKey = useMemo(
    () => buildSourceKey({ inputMode, caseIdValue, textValue }),
    [inputMode, caseIdValue, textValue],
  );

  useEffect(() => {
    setMessages([]);
    setQuestion("");
    setError(null);
  }, [sourceKey]);

  async function handleAsk() {
    if (!question.trim() || disabled) return;
    const q = question.trim();

    setError(null);
    setIsLoading(true);
    setMessages((prev) => [...prev, { role: "user", content: q }]);
    setQuestion("");

    try {
      const payload = { question: q, top_k_context: 4 };
      if (inputMode === "caseId") {
        payload.case_id = caseIdValue;
      } else {
        payload.text = textValue;
      }

      const response = await chatWithCase(payload);
      const rag = response?.data?.rag;
      const answer = rag?.answer || "I couldn't generate an answer for that question.";
      const citations = rag?.citations || [];

      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: answer,
          confidence: rag?.confidence,
          citations,
        },
      ]);
    } catch (err) {
      setError(err.message || "Failed to get an answer from case chat.");
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content:
            "I couldn't answer right now. Please retry your question in a moment.",
          citations: [],
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <section className="rounded-2xl border border-[var(--color-border-subtle)] bg-[var(--color-surface-panel)] p-6 shadow-[var(--shadow-sm-subtle)]">
      <div className="mb-2 flex items-center gap-2">
        <MessageSquare className="h-4 w-4 text-[var(--color-text-tertiary)]" />
        <h3 className="font-serif text-2xl text-[var(--color-text-primary)]">Chat with this Case</h3>
        <span className="ml-1 rounded-full border border-amber-300 bg-amber-50 px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider text-amber-700">
          Beta
        </span>
        <span
          className="inline-flex items-center text-[var(--color-text-tertiary)]"
          title="Beta feature: answers may be incomplete or occasionally inaccurate while we improve it."
          aria-label="Beta feature info"
        >
          <Info className="h-4 w-4" />
        </span>
      </div>

      <p className="mb-3 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-800">
        This feature is in beta and may not always work perfectly.
      </p>

      <p className="mb-4 text-sm text-[var(--color-text-secondary)]">
        Ask specific questions like <em>"What was the appellant's main argument?"</em> or <em>"What relief was granted?"</em>
      </p>

      <div className="mb-4 max-h-72 space-y-3 overflow-y-auto rounded-xl border border-[var(--color-border-subtle)] bg-[var(--color-surface-base)] p-3">
        {messages.length === 0 ? (
          <p className="text-sm text-[var(--color-text-tertiary)]">No questions yet. Start by asking one below.</p>
        ) : (
          messages.map((message, idx) => (
            <div
              key={`${message.role}-${idx}`}
              className={`rounded-lg px-3 py-2 text-sm ${
                message.role === "user"
                  ? "ml-10 bg-[var(--color-surface-elevated)] text-[var(--color-text-primary)]"
                  : "mr-10 border border-[var(--color-border-subtle)] bg-white text-[var(--color-text-primary)]"
              }`}
            >
              <p>{message.content}</p>
              {message.role === "assistant" && message.confidence && (
                <p className="mt-2 text-[10px] uppercase tracking-wider text-[var(--color-text-tertiary)]">
                  Confidence: {message.confidence}
                </p>
              )}
              {message.role === "assistant" && message.citations?.length > 0 && (
                <ul className="mt-2 list-disc space-y-1 pl-4 text-xs text-[var(--color-text-secondary)]">
                  {message.citations.slice(0, 2).map((citation) => (
                    <li key={`${idx}-${citation.rank}`}>{citation.text}</li>
                  ))}
                </ul>
              )}
            </div>
          ))
        )}
      </div>

      {error && (
        <p className="mb-3 rounded-lg border border-[var(--color-status-rejected-border)] bg-[var(--color-status-rejected-bg)] px-3 py-2 text-xs text-[var(--color-status-rejected-text)]">
          {error}
        </p>
      )}

      <div className="flex gap-2">
        <input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              e.preventDefault();
              handleAsk();
            }
          }}
          disabled={disabled || isLoading}
          placeholder="Ask a question about this case..."
          className="flex-1 rounded-xl border border-[var(--color-border-subtle)] bg-[var(--color-surface-base)] px-3 py-2 text-sm"
        />
        <button
          type="button"
          onClick={handleAsk}
          disabled={disabled || isLoading || !question.trim()}
          className="inline-flex items-center gap-1 rounded-xl bg-[var(--color-text-primary)] px-4 py-2 text-sm font-semibold text-[var(--color-surface-panel)] disabled:opacity-50"
        >
          {isLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
          Ask
        </button>
      </div>
    </section>
  );
}
