import { TextSelect, Quote, FileSearch } from "lucide-react";

function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function highlightSentenceText(text, keyTerms) {
  if (!text || !keyTerms?.length) return text;

  const uniqueTerms = [...new Set(keyTerms.map((term) => String(term).trim().toLowerCase()))]
    .filter(Boolean)
    .sort((a, b) => b.length - a.length);

  if (!uniqueTerms.length) return text;

  const regex = new RegExp(`(${uniqueTerms.map(escapeRegExp).join("|")})`, "gi");
  const chunks = String(text).split(regex);

  return chunks.map((chunk, index) => {
    const normalized = chunk.toLowerCase();
    const isTerm = uniqueTerms.includes(normalized);
    if (!isTerm) {
      return <span key={`plain-${index}`}>{chunk}</span>;
    }
    return (
      <mark
        key={`mark-${index}`}
        title={`Key Term: ${chunk}`}
        className="rounded bg-yellow-200/40 px-1 py-0.5 text-[var(--color-text-primary)] font-medium cursor-help"
      >
        {chunk}
      </mark>
    );
  });
}

function getPrimaryEvidenceScore(item) {
  return Math.max(
    Number(item?.accepted_evidence ?? 0),
    Number(item?.rejected_evidence ?? 0),
  );
}

function SentenceGroup({ title, toneClass, items, keyTerms, isAccepted }) {     
  const maxScore = items?.length
    ? Math.max(...items.map((item) => getPrimaryEvidenceScore(item)))
    : 0;

  if (!items?.length) return null;

  return (
    <div className="mt-6">
      <h4 className="flex items-center mb-4 text-xs font-semibold tracking-widest uppercase text-[var(--color-text-tertiary)] gap-2">                                   {isAccepted ? <span className="w-2 h-2 rounded-full bg-[var(--color-status-accepted-text)]" /> : <span className="w-2 h-2 rounded-full bg-[var(--color-status-rejected-text)]" />}                                                              {title}
      </h4>
      <div className="space-y-4">
        {items.map((item, index) => {
          const score = getPrimaryEvidenceScore(item);
          const relativeScore = maxScore ? score / maxScore : 0;

          return (
            <article
              key={`${title}-${index}-${item.start_char}`}
              title={isAccepted ? "Strong evidence supporting case acceptance" : "Strong evidence supporting case rejection"}
              className={`rounded-xl border p-5 transition-shadow cursor-help hover:shadow-[var(--shadow-md-subtle)] ${toneClass}`}                                                     >
              <div className="flex gap-4">
                <div className="pt-1 shrink-0">
                  <Quote className="w-4 h-4 opacity-40" />
                </div>
                <div className="flex-1">
                  <p className="text-[1.05rem] leading-relaxed text-[var(--color-text-primary)]">                                                                                   {highlightSentenceText(item.text, keyTerms)}
                  </p>

                  {/* Subtle bar indicator instead of explicit text label */}   
                  <div className="flex items-center mt-4 gap-3">
                     <div className="flex-1 h-1 overflow-hidden rounded-full bg-black/5">                                                                                               <div
                          className="h-full rounded-full bg-black/20"
                          style={{ width: `${Math.max(5, relativeScore * 100)}%` }}                                                                                                     />
                     </div>
                     <span className="text-[0.65rem] uppercase tracking-widest text-[var(--color-text-tertiary)] font-mono">                                                            {(relativeScore * 100).toFixed(0)}% Str
                     </span>
                  </div>
                </div>
              </div>
            </article>
          );
        })}
      </div>
    </div>
  );
}

export function EvidenceSentencesCard({ explanation, onViewDocument }) {        
  const sentenceCount = explanation?.sentence_evidence?.sentence_count ?? 0;    
  const keyTerms = [
    ...(explanation?.top_term_contributions?.accepted ?? []),
    ...(explanation?.top_term_contributions?.rejected ?? []),
  ]
    .map((item) => item?.term)
    .filter(Boolean)
    .slice(0, 18);

  const supporting = explanation?.sentence_evidence?.supporting ?? [];
  const opposing = explanation?.sentence_evidence?.opposing ?? [];

  return (
    <section className="rounded-2xl border border-[var(--color-border-subtle)] bg-[var(--color-surface-panel)] p-8 shadow-[var(--shadow-sm-subtle)]">                 <div className="flex items-center mb-2 gap-2">
        <TextSelect className="h-4 w-4 text-[var(--color-text-tertiary)]" />    
        <p className="text-xs font-semibold tracking-widest uppercase text-[var(--color-text-secondary)]">                                                                Key Passages
        </p>
        <span className="ml-auto rounded-full bg-[var(--color-surface-elevated)] px-2.5 py-0.5 text-[0.65rem] font-bold uppercase tracking-wider text-[var(--color-text-primary)]">                                                                       {sentenceCount} Found
        </span>
      </div>

      <div className="flex flex-col justify-between sm:flex-row sm:items-center gap-4"><h3 className="font-serif text-3xl text-[var(--color-text-primary)]">Passages that influenced prediction</h3>{onViewDocument && (<button onClick={onViewDocument} className="flex items-center gap-2 rounded-lg bg-[var(--color-surface-base)] px-4 py-2 text-sm font-semibold text-[var(--color-text-primary)] border border-[var(--color-border-strong)] hover:bg-[var(--color-surface-elevated)] transition-colors shadow-sm"><FileSearch className="w-4 h-4" />View in Document</button>)}</div>                                                                     
      <div className="mt-6 space-y-8">
        <SentenceGroup
          title="Supports Acceptance"
          toneClass="bg-[var(--color-status-accepted-bg)]/30 border-[var(--color-status-accepted-border)]/50"                                                             items={supporting}
          keyTerms={keyTerms}
          isAccepted={true}
        />
        <SentenceGroup
          title="Supports Rejection"
          toneClass="bg-[var(--color-status-rejected-bg)]/30 border-[var(--color-status-rejected-border)]/50"                                                             items={opposing}
          keyTerms={keyTerms}
          isAccepted={false}
        />
      </div>
    </section>
  );
}
