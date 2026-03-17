import { useState } from "react";
import { FileText, Hash, Play, Trash2, Terminal, ChevronDown, ChevronUp } from "lucide-react";

function TabButton({ active, label, icon: Icon, onClick }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={"flex items-center justify-center gap-2 rounded-lg border px-4 py-3 sm:py-2.5 text-sm font-semibold transition-all " + (
        active
          ? "border-[var(--color-text-primary)] bg-[var(--color-text-primary)] text-[var(--color-surface-panel)] shadow-md"
          : "border-[var(--color-border-subtle)] bg-[var(--color-surface-base)] text-[var(--color-text-secondary)] hover:bg-[var(--color-surface-elevated)] hover:border-[var(--color-border-strong)]"
      )}
    >
      <Icon className="h-4 w-4" />
      {label}
    </button>
  );
}

export function InputPanel({ history, onRestoreHistory, onClearHistory,
  inputMode,
  setInputMode,
  textValue,
  setTextValue,
  caseIdValue,
  setCaseIdValue,
  isLoading,
  errorMessage,
  sampleCases,
  onAnalyze,
  onClear,
  onSampleSelect,
}) {
  const isTextMode = inputMode === "text";
  const [showAllHistory, setShowAllHistory] = useState(false);
  const displayedHistory = history ? (showAllHistory ? history : history.slice(0, 3)) : [];
  const textChars = textValue.length;
  const canAnalyze = isTextMode ? textValue.trim().length > 0 : caseIdValue.trim().length > 0;

  function getCharCountStatus(chars) {
    if (chars === 0) return { color: "text-[var(--color-text-tertiary)]", label: "0 chars" };
    if (chars > 400000) return { color: "text-amber-600", label: `${chars.toLocaleString()} chars (Very long)` };
    return { color: "text-[var(--color-status-accepted-text)]", label: `${chars.toLocaleString()} chars` };
  }

  const charStatus = getCharCountStatus(textChars);
  
  return (
    <section className="rounded-2xl border border-[var(--color-border-subtle)] bg-[var(--color-surface-panel)] p-6 shadow-[var(--shadow-lg-subtle)]">
      <div className="flex items-center gap-2 mb-4">
        <Terminal className="h-4 w-4 text-[var(--color-text-tertiary)]" />
        <h2 className="font-serif text-2xl text-[var(--color-text-primary)]">
          Input Console
        </h2>
      </div>

      <div className="grid grid-cols-2 gap-2 p-1 bg-[var(--color-surface-elevated)] rounded-xl border border-[var(--color-border-subtle)]">
        <TabButton
          active={isTextMode}
          label="Search / Text"
          icon={FileText}
          onClick={() => setInputMode("text")}
        />
        <TabButton
          active={!isTextMode}
          label="Case ID"
          icon={Hash}
          onClick={() => setInputMode("caseId")}
        />
      </div>

      {isTextMode ? (
        <div className="mt-6 flex flex-col h-[320px]">
          <div className="flex items-center justify-between mb-2 px-1">
            <label className="text-xs font-semibold uppercase tracking-widest text-[var(--color-text-secondary)]">
              Search Query or Case Text
            </label>
            <span className={`font-mono text-xs ${charStatus.color}`}>{charStatus.label}</span>
          </div>
          <textarea
            value={textValue}
            onChange={(event) => setTextValue(event.target.value)}
            placeholder="Type a legal query (e.g. 'property disputes'), factual summary, or paste full judgment text..."
            className="flex-1 w-full resize-none rounded-xl border border-[var(--color-border-subtle)] bg-[var(--color-surface-base)] px-4 py-4 text-sm leading-relaxed text-[var(--color-text-primary)] placeholder-[var(--color-text-tertiary)] transition-colors focus:border-[var(--color-text-primary)] focus:outline-none focus:ring-1 focus:ring-[var(--color-text-primary)]"
          />
        </div>
      ) : (
        <div className="mt-6 min-h-[320px] flex flex-col">
          <label className="text-xs font-semibold uppercase tracking-widest text-[var(--color-text-secondary)] mb-2 px-1">
            Dataset Case ID
          </label>
          <input
            value={caseIdValue}
            onChange={(event) => setCaseIdValue(event.target.value)}
            placeholder="Example: 1980_211"
            className="w-full rounded-xl border border-[var(--color-border-subtle)] bg-[var(--color-surface-base)] px-4 py-4 sm:py-3.5 text-sm font-medium text-[var(--color-text-primary)] placeholder-[var(--color-text-tertiary)] transition-colors focus:border-[var(--color-text-primary)] focus:outline-none focus:ring-1 focus:ring-[var(--color-text-primary)]"
          />

          {history && history.length > 0 ? (
            <div className="mt-8 flex-1">
              <div className="flex items-center justify-between mb-3 px-1">
                <p className="text-xs font-semibold uppercase tracking-widest text-[var(--color-text-secondary)]">
                  Recent Analyses
                </p>
                <button 
                  onClick={onClearHistory}
                  className="text-[10px] uppercase font-bold text-[var(--color-text-tertiary)] hover:text-red-500 transition-colors"
                  title="Clear all history"
                >
                  Clear History
                </button>
              </div>
              <div className="flex flex-col gap-2">
                {displayedHistory.map((entry) => (
                  <button
                    key={entry.id}
                    type="button"
                    onClick={() => onRestoreHistory(entry.fullResult, entry.type, entry.query)}
                    className="flex items-center justify-between w-full rounded-lg border border-[var(--color-border-subtle)] bg-[var(--color-surface-base)] px-4 py-2 text-sm font-medium text-[var(--color-text-primary)] transition-all hover:border-[var(--color-border-strong)] hover:bg-[var(--color-surface-elevated)] active:scale-[0.98] text-left"
                  >
                    <div className="flex flex-col flex-1 min-w-0 pr-2">
                       <span className="truncate max-w-[200px] font-mono">{entry.query}</span>
                       <span className="text-[10px] text-[var(--color-text-tertiary)] uppercase tracking-wider">{new Date(entry.timestamp).toLocaleDateString()} &bull; {entry.type}</span>
                    </div>
                    <div className={`shrink-0 w-2 h-2 rounded-full ${entry.verdict === "accepted" ? "bg-[var(--color-status-accepted-text)]" : entry.verdict === "rejected" ? "bg-[var(--color-status-rejected-text)]" : "bg-gray-400"}`}></div>
                  </button>
                ))}
                {history.length > 3 && (
                  <button
                    onClick={() => setShowAllHistory(!showAllHistory)}
                    className="flex justify-center items-center gap-1 w-full py-1.5 mt-1 text-xs font-medium text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] transition-colors rounded hover:bg-[var(--color-surface-elevated)]"
                  >
                    {showAllHistory ? (
                      <><ChevronUp className="w-3 h-3" /> Show Less</>
                    ) : (
                      <><ChevronDown className="w-3 h-3" /> Show {history.length - 3} More</>
                    )}
                  </button>
                )}
              </div>
            </div>
          ) : (
            <div className="mt-8 flex-1">
              <p className="text-xs font-semibold uppercase tracking-widest text-[var(--color-text-secondary)] mb-3 px-1">
                Quick Samples
              </p>
              <div className="flex flex-col gap-2">
                {sampleCases.map((caseId) => (
                  <button
                    key={caseId}
                    type="button"
                    onClick={() => onSampleSelect(caseId)}
                    className="flex items-center justify-between w-full rounded-lg border border-[var(--color-border-subtle)] bg-[var(--color-surface-base)] px-4 py-3 sm:py-2.5 text-sm font-medium text-[var(--color-text-primary)] transition-all hover:border-[var(--color-border-strong)] hover:bg-[var(--color-surface-elevated)] active:scale-[0.98]"
                  >
                    <span className="font-mono">{caseId}</span>
                    <span className="text-xs text-[var(--color-text-tertiary)] hover:text-[var(--color-text-secondary)]">Use &rarr;</span>
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {errorMessage ? (
        <div className="mt-4 rounded-xl border border-[var(--color-status-rejected-border)] bg-[var(--color-status-rejected-bg)] px-4 py-3">
          <p className="text-sm font-medium text-[var(--color-status-rejected-text)]">
            {errorMessage}
          </p>
          {!isTextMode && (
            <p className="mt-1 text-xs text-[var(--color-status-rejected-text)] opacity-80">
              Tip: Check the case ID format (example: <span className="font-mono">1983_326</span>) or choose one from Quick Samples.
            </p>
          )}
        </div>
      ) : null}

      

      <div className="mt-6 pt-6 border-t border-[var(--color-border-subtle)] flex flex-col gap-3 sm:flex-row">
        <button
          type="button"
          onClick={onClear}
          className="flex items-center justify-center gap-2 rounded-xl border border-[var(--color-border-subtle)] bg-[var(--color-surface-base)] px-5 py-4 sm:py-3.5 text-sm font-semibold text-[var(--color-text-secondary)] transition-colors hover:bg-[var(--color-surface-elevated)] hover:text-[var(--color-text-primary)]"
        >
          <Trash2 className="h-4 w-4" />
          Clear
        </button>
        <button
          type="button"
          onClick={onAnalyze}
          disabled={!canAnalyze || isLoading}
          className="flex-1 flex items-center justify-center gap-2 rounded-xl bg-[var(--color-text-primary)] px-5 py-4 sm:py-3.5 text-sm font-bold text-[var(--color-surface-panel)] shadow-md transition-all hover:bg-black active:scale-[0.98] disabled:pointer-events-none disabled:opacity-50"
        >
          {isLoading ? (
             <span className="relative flex h-4 w-4">
                <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-white opacity-75"></span>
                <span className="relative inline-flex h-4 w-4 rounded-full bg-white opacity-90"></span>
             </span>
          ) : (
            <Play className="h-4 w-4 fill-current" />
          )}
            {isLoading ? "Processing..." : (isTextMode && textChars > 0 && textChars < 300 ? "Search Database" : "Run Analysis")}
        </button>
      </div>
    </section>
  );
}







