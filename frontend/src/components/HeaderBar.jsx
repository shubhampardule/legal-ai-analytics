import { Brain, Wand2, Database, AlertCircle, Scale, LibrarySquare } from "lucide-react";

function StatusPill({ apiHealthy }) {
  if (apiHealthy === null) {
    return (
      <span className="flex items-center gap-1.5 rounded-full border border-[var(--color-border-subtle)] px-2.5 py-1 text-[0.68rem] font-bold uppercase tracking-wider text-[var(--color-text-secondary)] shadow-sm bg-white/50">
        <span className="relative flex h-2 w-2">
          <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-gray-400 opacity-75"></span>
          <span className="relative inline-flex h-2 w-2 rounded-full bg-gray-500"></span>
        </span>
        Connecting
      </span>
    );
  }

  return apiHealthy ? (
    <span className="flex items-center gap-1.5 rounded-full border border-[var(--color-status-accepted-border)] px-2.5 py-1 text-[0.68rem] font-bold uppercase tracking-wider text-[var(--color-status-accepted-text)] shadow-sm bg-[var(--color-status-accepted-bg)]">
      <span className="relative flex h-2 w-2">
        <span className="relative inline-flex h-2 w-2 rounded-full bg-[var(--color-status-accepted-text)]"></span>
      </span>
      System Active
    </span>
  ) : (
    <span className="flex items-center gap-1.5 rounded-full border border-[var(--color-status-rejected-border)] px-2.5 py-1 text-[0.68rem] font-bold uppercase tracking-wider text-[var(--color-status-rejected-text)] shadow-sm bg-[var(--color-status-rejected-bg)]">
      <AlertCircle className="h-3 w-3" />
      System Offline
    </span>
  );
}

export function HeaderBar({ apiHealthy, apiMeta, onBrowseDatabase }) {
  return (
    <header className="flex w-full items-center justify-between gap-4">
      <div className="flex items-center gap-4 sm:gap-6">
        <div className="flex items-center gap-2.5">
           <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-[var(--color-text-primary)] text-[var(--color-surface-panel)] shadow-sm">
              <Scale className="h-4 w-4" />
           </div>
            <h1 className="font-serif text-2xl font-bold leading-none text-[var(--color-text-primary)] tracking-tight">
              Legal Core
            </h1>
        </div>

        <div className="hidden h-5 w-px bg-[var(--color-border-subtle)] sm:block"></div>
        <StatusPill apiHealthy={apiHealthy} />
      </div>

      <div className="flex items-center gap-3 text-[0.68rem] font-mono text-[var(--color-text-secondary)]">
         {onBrowseDatabase && (
             <button 
                onClick={onBrowseDatabase}
                className="flex items-center gap-1.5 bg-[var(--color-surface-base)] hover:bg-[var(--color-surface-elevated)] hover:text-[var(--color-text-primary)] px-3 py-1.5 rounded-lg border border-[var(--color-border-strong)] transition-all mr-2 shadow-sm"
             >
                 <LibrarySquare className="h-4 w-4" />
                 <span className="font-sans font-semibold text-xs">Cases DB</span>
             </button>
         )}
         <div className="hidden xl:flex items-center gap-1.5 bg-[var(--color-surface-elevated)] px-2.5 py-1 rounded border border-[var(--color-border-subtle)]">
            <Brain className="h-3 w-3 text-[var(--color-text-tertiary)]" />
            {apiMeta?.prediction_model ?? "baseline_tfidf_logreg"}
         </div>
         <div className="hidden xl:flex items-center gap-1.5 bg-[var(--color-surface-elevated)] px-2.5 py-1 rounded border border-[var(--color-border-subtle)]">
            <Wand2 className="h-3 w-3 text-[var(--color-text-tertiary)]" />
            {apiMeta?.explanation_method ?? "linear_feature_contribution"}
         </div>
         <div className="hidden xl:flex items-center gap-1.5 bg-[var(--color-surface-elevated)] px-2.5 py-1 rounded border border-[var(--color-border-subtle)]">
            <Database className="h-3 w-3 text-[var(--color-text-tertiary)]" />
            {apiMeta?.similarity_model ?? "all-MiniLM-L6-v2"}
         </div>
      </div>
    </header>
  );
}
