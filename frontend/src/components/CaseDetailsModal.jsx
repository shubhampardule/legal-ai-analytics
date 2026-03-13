import { useState, useEffect } from "react";
import { X, FileText, Loader2 } from "lucide-react";
import { getCaseDetails } from "../api/client";

function HighlightedDocument({ text, highlights }) {
  if (!text) return null;
  if (!highlights || (!highlights.supporting?.length && !highlights.opposing?.length)) {
    return <>{text}</>;
  }

  const markers = [];

  const findMatches = (sentences, type) => {
    sentences.forEach(sentence => {
      if (!sentence) return;
      
      let startIndex = 0;
      let matched = false;
      while ((startIndex = String(text).indexOf(sentence, startIndex)) !== -1) {
        markers.push({ start: startIndex, end: startIndex + sentence.length, type });
        startIndex += sentence.length;
        matched = true;
      }
      
      if (!matched) {
         const normSentence = sentence.replace(/\s+/g, ' ').trim();
         if (normSentence.length < 10) return;
         
         const escapedWords = normSentence.split(' ').map(w => w.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
         const fuzzyPattern = new RegExp(escapedWords.join('\\s+'), 'g');
         
         let match;
         while ((match = fuzzyPattern.exec(text)) !== null) {
            markers.push({ start: match.index, end: match.index + match[0].length, type });
         }
      }
    });
  };

  findMatches(highlights.supporting || [], 'supporting');
  findMatches(highlights.opposing || [], 'opposing');

  if (markers.length === 0) {
    return <>{text}</>;
  }

  markers.sort((a, b) => a.start - b.start);

  const resolvedMarkers = [];
  let currentEnd = 0;
  for (const m of markers) {
    if (m.start >= currentEnd) {
      resolvedMarkers.push(m);
      currentEnd = m.end;
    }
  }

  const elements = [];
  let lastIndex = 0;

  resolvedMarkers.forEach((m, idx) => {
    if (m.start > lastIndex) {
      elements.push(<span key={"text-" + idx}>{text.slice(lastIndex, m.start)}</span>);
    }
    const isSupport = m.type === 'supporting';
    const highlightClass = isSupport 
        ? "bg-[var(--color-status-accepted-bg)] text-[var(--color-status-accepted-text)] border border-[var(--color-status-accepted-border)]" 
        : "bg-[var(--color-status-rejected-bg)] text-[var(--color-status-rejected-text)] border border-[var(--color-status-rejected-border)]";
    elements.push(
      <span
        key={"mark-" + idx}
        className={"px-1 py-0.5 rounded-sm font-medium transition-colors cursor-help " + highlightClass}
        title={isSupport ? "Supporting Evidence" : "Opposing Evidence"}
      >
        {text.slice(m.start, m.end)}
      </span>
    );
    lastIndex = m.end;
  });

  if (lastIndex < text.length) {
    elements.push(<span key="text-last">{text.slice(lastIndex)}</span>);
  }

  return <>{elements}</>;
}

export function CaseDetailsModal({ caseId, rawText, highlights, onClose }) {
  const [caseData, setCaseData] = useState(null);
  const [loading, setLoading] = useState(!rawText);
  const [error, setError] = useState(null);

  useEffect(() => {
    async function fetchCase() {
      if (rawText) {
         setCaseData({ clean_text: rawText, clean_char_length: rawText.length });
         setLoading(false);
         return;
      }
      try {
        setLoading(true);
        const data = await getCaseDetails(caseId);
        setCaseData(data.data.case);
        setError(null);
      } catch (err) {
        setError(err.message || "Failed to load case details.");
      } finally {
        setLoading(false);
      }
    }
    fetchCase();
  }, [caseId, rawText]);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm animate-in fade-in duration-200">
      <div
        className="relative flex flex-col w-full max-w-4xl max-h-[90vh] bg-[var(--color-surface-base)] rounded-2xl shadow-2xl border border-[var(--color-border-strong)] overflow-hidden"
        role="dialog"
      >
        <div className="flex items-center justify-between px-6 py-4 border-b border-[var(--color-border-subtle)] bg-[var(--color-surface-panel)]">
          <div className="flex items-center gap-3">
            <FileText className="w-5 h-5 text-[var(--color-text-secondary)]" />
            <h2 className="text-xl font-semibold font-serif text-[var(--color-text-primary)]">
              {caseId ? "Case Details #" + caseId : "Analyzed Document"}
            </h2>
          </div>
          <button
            onClick={onClose}
            className="p-2 rounded-full hover:bg-[var(--color-surface-elevated)] transition-colors text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)]"
            aria-label="Close dialog"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-6 lg:p-8 bg-[var(--color-surface-base)]">
          {loading ? (
            <div className="flex flex-col items-center justify-center h-64 gap-4 text-[var(--color-text-secondary)]">
              <Loader2 className="w-8 h-8 animate-spin" />
              <p>Loading full case history...</p>
            </div>
          ) : error ? (
            <div className="flex items-center justify-center h-64 text-[var(--color-status-rejected-text)] p-4 bg-[var(--color-status-rejected-bg)] rounded-xl border border-[var(--color-status-rejected-border)]">
              <p>{error}</p>
            </div>
          ) : caseData ? (
            <div className="space-y-6">
              <div className="flex flex-wrap items-center justify-between gap-4">
                <div className="flex items-center gap-4">
                    {caseData.label_name && (
                        <span
                            className={"rounded-full px-3 py-1 text-xs font-bold uppercase tracking-wider " + (
                                caseData.label_name === "accepted"
                                ? "bg-[var(--color-status-accepted-bg)] text-[var(--color-status-accepted-text)] border border-[var(--color-status-accepted-border)]"
                                : "bg-[var(--color-status-rejected-bg)] text-[var(--color-status-rejected-text)] border border-[var(--color-status-rejected-border)]"
                            )}
                        >
                            Status: {caseData.label_name === "accepted" ? "Accepted" : "Rejected"}
                        </span>
                    )}
                    <span className="text-sm font-medium text-[var(--color-text-secondary)]">
                        Word Count: {caseData.clean_char_length} characters
                    </span>
                </div>
                {highlights && (
                   <div className="flex items-center gap-3 text-xs font-semibold uppercase tracking-widest">
                       <div className="flex items-center gap-1.5"><span className="w-3 h-3 rounded bg-[var(--color-status-accepted-bg)] border border-[var(--color-status-accepted-border)] inline-block"></span> Supports</div>
                       <div className="flex items-center gap-1.5"><span className="w-3 h-3 rounded bg-[var(--color-status-rejected-bg)] border border-[var(--color-status-rejected-border)] inline-block"></span> Opposes</div>
                   </div>
                )}
              </div>

              <div className="prose prose-sm max-w-none text-[var(--color-text-primary)] leading-relaxed bg-[var(--color-surface-panel)] p-6 rounded-xl border border-[var(--color-border-subtle)] whitespace-pre-wrap">
                <HighlightedDocument text={caseData.clean_text} highlights={highlights} />
              </div>
            </div>
          ) : null}
        </div>

        <div className="border-t border-[var(--color-border-subtle)] bg-[var(--color-surface-panel)] px-6 py-4 flex justify-end">
            <button
                onClick={onClose}
                className="px-6 py-2 rounded-full font-medium text-sm border border-[var(--color-border-strong)] hover:bg-[var(--color-surface-elevated)] transition-colors"
            >
                Close
            </button>
        </div>
      </div>
    </div>
  );
}
