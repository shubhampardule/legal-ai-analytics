import { AlertCircle, RotateCcw } from "lucide-react";

export function ErrorState({ message, onRetry }) {
  return (
    <div className="flex h-full w-full flex-col items-center justify-center rounded-2xl border border-dashed border-[var(--color-status-rejected-border)] bg-[var(--color-status-rejected-bg)]/30 p-12 text-center animate-in fade-in duration-500">
      <div className="flex h-20 w-20 items-center justify-center rounded-full bg-[var(--color-status-rejected-bg)] text-[var(--color-status-rejected-text)] shadow-inner mb-6">
        <AlertCircle className="h-10 w-10" />
      </div>
      <h2 className="font-serif text-2xl text-[var(--color-text-primary)]">
        Analysis Failed
      </h2>
      <p className="mt-3 max-w-md text-sm leading-relaxed text-[var(--color-status-rejected-text)]">
        {message}
      </p>
      <button
        onClick={onRetry}
        className="mt-8 flex items-center justify-center gap-2 rounded-xl bg-[var(--color-surface-panel)] px-6 py-3 text-sm font-semibold text-[var(--color-text-primary)] border border-[var(--color-border-strong)] hover:bg-[var(--color-surface-elevated)] hover:text-black transition-colors shadow-sm"
      >
        <RotateCcw className="h-4 w-4" />
        Retry Analysis
      </button>
    </div>
  );
}
