import { ClipboardList, Download } from "lucide-react";

function resolveKeySupportingTerm(prediction, explanation) {
  const predictedAccepted = prediction?.predicted_label === "accepted";
  const termGroups = explanation?.top_term_contributions;
  const primaryGroup = predictedAccepted ? termGroups?.accepted : termGroups?.rejected;
  const fallbackGroup = predictedAccepted ? termGroups?.rejected : termGroups?.accepted;
  
  const primaryTerm = Array.isArray(primaryGroup) && primaryGroup.length > 0 ? primaryGroup[0]?.term : null;
  const fallbackTerm = Array.isArray(fallbackGroup) && fallbackGroup.length > 0 ? fallbackGroup[0]?.term : null;
  
  return primaryTerm || fallbackTerm || "No dominant term identified";
}

export function AnalysisSummaryCard({ summary, prediction, explanation }) {
  if (!summary) {
    return null;
  }

  const keySupportingTerm = resolveKeySupportingTerm(prediction, explanation);
  const evidenceCount = Number(summary.evidence_count ?? 0);
  const similarCaseQuality = summary.similar_case_quality ?? "Unknown";

  return (
    <section className="rounded-2xl border border-[var(--color-border-subtle)] bg-[var(--color-surface-panel)] p-8 shadow-[var(--shadow-md-subtle)] print:shadow-none print:p-0 print:border-none">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
           <ClipboardList className="h-4 w-4 text-[var(--color-text-tertiary)]" />
           <p className="text-xs font-semibold uppercase tracking-widest text-[var(--color-text-secondary)]">
              Executive Summary
           </p>
        </div>
        <button 
          onClick={() => window.print()}
          className="flex items-center gap-2 rounded-lg bg-[var(--color-surface-base)] px-3 py-1.5 text-xs font-semibold text-[var(--color-text-secondary)] border border-[var(--color-border-subtle)] hover:bg-[var(--color-surface-elevated)] transition-colors print:hidden"
          title="Print to PDF"
        >
          <Download className="h-3.5 w-3.5" />
          Download Report
        </button>
      </div>

      <h2 className="mt-2 font-serif text-3xl text-[var(--color-text-primary)]">{summary.verdict}</h2>
      
      <p className="mt-4 max-w-3xl text-sm leading-relaxed text-[var(--color-text-secondary)]">
        {summary.sentence}
      </p>

      <div className="mt-6 rounded-xl bg-[var(--color-surface-base)] border border-[var(--color-border-subtle)] p-5">
        <ul className="space-y-3 text-sm text-[var(--color-text-primary)]">
          <li className="flex flex-wrap gap-2">
            <span className="text-[var(--color-text-tertiary)] min-w-[140px]">Key Element:</span>
            <span className="font-semibold">{keySupportingTerm}</span>
          </li>
          <li className="flex flex-wrap gap-2">
            <span className="text-[var(--color-text-tertiary)] min-w-[140px]">Evidence Mapped:</span>
            <span className="font-semibold">{evidenceCount} passages</span>
          </li>
          <li className="flex flex-wrap gap-2">
            <span className="text-[var(--color-text-tertiary)] min-w-[140px]">Precedent match:</span>
            <span className="font-semibold">{similarCaseQuality}</span>
          </li>
        </ul>
      </div>
    </section>
  );
}

