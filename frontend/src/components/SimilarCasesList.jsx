import { useState } from "react";
import { History, ChevronRight, ChevronDown } from "lucide-react";
import { CaseDetailsModal } from "./CaseDetailsModal";

function formatPreview(text, maxChars = 220) {
  const value = String(text ?? "").trim();
  if (value.length <= maxChars) return value;
  return value.slice(0, maxChars).trimEnd() + "...";
}

export function SimilarCasesList({ retrieval, filters, onFilterChange, onApplyFilters, onResetFilters }) {
  const results = retrieval?.results ?? [];
  const [activeCaseId, setActiveCaseId] = useState(null);
  const [showAll, setShowAll] = useState(false);

  if (!results.length) return null;

  const displayedResults = showAll ? results : results.slice(0, 6);

  return (
    <section className="rounded-2xl border border-[var(--color-border-subtle)] bg-[var(--color-surface-panel)] p-8 shadow-[var(--shadow-sm-subtle)]">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between mb-8">
        <div>
          <div className="flex items-center gap-2 mb-2">
            <History className="h-4 w-4 text-[var(--color-text-tertiary)]" />
            <p className="text-xs font-semibold uppercase tracking-widest text-[var(--color-text-secondary)]">
              Precedent Analysis
            </p>
          </div>
          <h3 className="font-serif text-3xl text-[var(--color-text-primary)]">
            Similar Historic Cases
          </h3>
        </div>

        <div className="rounded-full bg-[var(--color-surface-elevated)] px-4 py-2 border border-[var(--color-border-subtle)]">
          <p className="text-xs font-medium text-[var(--color-text-secondary)]">
            {retrieval?.disclaimer ?? "Similarity does not guarantee the same legal outcome."}
          </p>
        </div>
      </div>

      <div className="mb-6 rounded-xl border border-[var(--color-border-subtle)] bg-[var(--color-surface-base)] p-4">
        <div className="grid gap-3 sm:grid-cols-4">
          <select
            value={filters?.outcome ?? ""}
            onChange={(e) => onFilterChange?.("outcome", e.target.value || null)}
            className="rounded-lg border border-[var(--color-border-subtle)] bg-[var(--color-surface-panel)] px-3 py-2 text-sm"
          >
            <option value="">All outcomes</option>
            <option value="accepted">Accepted only</option>
            <option value="rejected">Rejected only</option>
          </select>

          <input
            type="number"
            placeholder="Year from"
            value={filters?.year_from ?? ""}
            onChange={(e) => onFilterChange?.("year_from", e.target.value ? Number(e.target.value) : null)}
            className="rounded-lg border border-[var(--color-border-subtle)] bg-[var(--color-surface-panel)] px-3 py-2 text-sm"
            min={1800}
            max={2100}
          />

          <input
            type="number"
            placeholder="Year to"
            value={filters?.year_to ?? ""}
            onChange={(e) => onFilterChange?.("year_to", e.target.value ? Number(e.target.value) : null)}
            className="rounded-lg border border-[var(--color-border-subtle)] bg-[var(--color-surface-panel)] px-3 py-2 text-sm"
            min={1800}
            max={2100}
          />

          <div className="flex gap-2">
            <button
              type="button"
              onClick={onApplyFilters}
              className="flex-1 rounded-lg bg-[var(--color-text-primary)] px-3 py-2 text-sm font-semibold text-[var(--color-surface-panel)]"
            >
              Apply
            </button>
            <button
              type="button"
              onClick={onResetFilters}
              className="rounded-lg border border-[var(--color-border-subtle)] bg-[var(--color-surface-panel)] px-3 py-2 text-sm"
            >
              Reset
            </button>
          </div>
        </div>
      </div>

      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
        {displayedResults.map((item) => {
          const isAccepted = item.label_name === "accepted";
          const similarityScore = Math.round(Number(item.similarity_score ?? 0) * 100);

          return (
            <article
              key={item.rank + "-" + item.case_id}
              className="flex flex-col rounded-xl border border-[var(--color-border-subtle)] bg-[var(--color-surface-base)] overflow-hidden transition-all hover:shadow-[var(--shadow-md-subtle)] hover:border-[var(--color-border-strong)]"
            >
              <div className="p-5 border-b border-[var(--color-border-subtle)] bg-[var(--color-surface-panel)]">
                <div className="flex items-start justify-between mb-4">
                  <div className="flex flex-col gap-1">
                    <span className="text-sm font-semibold text-[var(--color-text-primary)]">
                      Case #{item.case_id}
                    </span>
                    <span className="text-xs font-mono text-[var(--color-text-tertiary)]">
                      Rank #{item.rank}
                    </span>
                  </div>
                  <span
                    className={`rounded-full px-2.5 py-1 text-[0.65rem] font-bold uppercase tracking-wider ${
                      isAccepted
                        ? "bg-[var(--color-status-accepted-bg)] text-[var(--color-status-accepted-text)] border border-[var(--color-status-accepted-border)]"
                        : "bg-[var(--color-status-rejected-bg)] text-[var(--color-status-rejected-text)] border border-[var(--color-status-rejected-border)]"
                    }`}
                  >
                    {isAccepted ? "Accepted" : "Rejected"}
                  </span>
                </div>

                <div className="flex items-center gap-3">
                  <div className="flex-1 h-1.5 overflow-hidden rounded-full bg-black/5">
                    <div
                      className="h-full rounded-full bg-current opacity-40 text-[var(--color-text-secondary)]"
                      style={{ width: Math.max(5, similarityScore) + "%" }}
                    />
                  </div>
                  <span className="text-xs font-semibold text-[var(--color-text-secondary)]">
                    {similarityScore}% Match
                  </span>
                </div>
              </div>

              <div className="p-5 flex-1 flex flex-col bg-[var(--color-surface-base)] relative">
                <p className="text-sm leading-relaxed text-[var(--color-text-secondary)] opacity-80 mb-4">
                  {formatPreview(item.preview_text)}
                </p>

                <button
                  type="button"
                  onClick={() => setActiveCaseId(item.case_id)}
                  className="mt-auto flex items-center gap-1 self-start rounded-full px-4 py-2 text-xs font-medium text-[var(--color-text-primary)] bg-[var(--color-surface-elevated)] transition-colors hover:bg-[var(--color-border-subtle)] border border-[var(--color-border-subtle)]"
                >
                  View detail <ChevronRight className="h-3 w-3" />
                </button>
              </div>
            </article>
          );
        })}
      </div>

      {!showAll && results.length > 6 && (
        <div className="mt-8 flex justify-center">
            <button
              onClick={() => setShowAll(true)}
              className="flex items-center gap-2 rounded-full border border-[var(--color-border-strong)] bg-[var(--color-surface-base)] px-6 py-2.5 text-sm font-semibold text-[var(--color-text-primary)] transition-colors hover:bg-[var(--color-surface-elevated)] hover:text-black shadow-sm"
            >
              Show all cases <ChevronDown className="h-4 w-4 opacity-70" />
            </button>
        </div>
      )}

      {activeCaseId && (
        <CaseDetailsModal
          caseId={activeCaseId}
          onClose={() => setActiveCaseId(null)}
        />
      )}
    </section>
  );
}
