import { ExternalLink } from "lucide-react";

export function Footer() {
  return (
    <footer className="mt-12 border-t border-[var(--color-border-subtle)] pb-8 pt-8 text-center text-sm text-[var(--color-text-tertiary)] print:hidden">
      <p className="mb-2">
        <strong className="font-semibold text-[var(--color-text-secondary)]">Disclaimer:</strong> This tool is for research purposes only and does not constitute legal advice.
      </p>
      <div className="flex items-center justify-center gap-3">
        <span>v0.1.0</span>
        <span className="text-[var(--color-border-strong)]">&bull;</span>
        <a
          href="https://github.com/shubhampardule"
          target="_blank"
          rel="noreferrer"
          className="flex items-center gap-1.5 text-[var(--color-primary-base)] hover:text-[var(--color-primary-hover)] underline underline-offset-4 decoration-[var(--color-primary-base)]/30 hover:decoration-[var(--color-primary-hover)] transition-colors"
        >
          @shubhampardule<ExternalLink className="h-3 w-3" />
        </a>
        <span className="text-[var(--color-border-strong)]">&bull;</span>
        <a href="https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2" target="_blank" rel="noreferrer" className="flex items-center gap-1.5 text-[var(--color-primary-base)] hover:text-[var(--color-primary-hover)] underline underline-offset-4 decoration-[var(--color-primary-base)]/30 hover:decoration-[var(--color-primary-hover)] transition-colors">
          Model Info<ExternalLink className="h-3 w-3" />
        </a>
      </div>
    </footer>
  );
}

