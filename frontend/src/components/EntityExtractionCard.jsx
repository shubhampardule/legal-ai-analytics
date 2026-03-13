import { Scale, Users, FileSignature } from "lucide-react";

export function EntityExtractionCard({ entities }) {
  const statutes = entities?.statutes || [];
  const judges = entities?.judges || [];
  
  if (statutes.length === 0 && judges.length === 0) {
     return null; // hide if empty
  }

  return (
    <section className="rounded-2xl border border-[var(--color-border-subtle)] bg-[var(--color-surface-panel)] p-8 shadow-[var(--shadow-sm-subtle)] h-full flex flex-col">
      <div className="flex items-center gap-2 mb-4">
        <FileSignature className="h-4 w-4 text-[var(--color-text-tertiary)]" />
        <h3 className="font-serif text-xl font-medium text-[var(--color-text-primary)]">Key Statutes & Entities</h3>
      </div>
      
      <p className="text-sm text-[var(--color-text-secondary)] mb-6">
        Laws and key figures extracted from the original case document.
      </p>

      <div className="space-y-6 flex-1">
        {statutes.length > 0 && (
            <div>
              <h4 className="text-xs font-semibold uppercase tracking-widest text-[var(--color-text-tertiary)] mb-3 flex items-center gap-2">
                 <Scale className="w-4 h-4" /> Relevant Statutes
              </h4>
              <div className="flex flex-wrap gap-2">
                 {statutes.map((statute, i) => (
                    <span key={`statute-${i}`} className="px-3 py-1.5 bg-blue-100/50 text-blue-900 border border-blue-200 rounded-md text-sm font-medium">
                       {statute}
                    </span>
                 ))}
              </div>
            </div>
        )}

        {judges.length > 0 && (
            <div>
              <h4 className="text-xs font-semibold uppercase tracking-widest text-[var(--color-text-tertiary)] mb-3 flex items-center gap-2">
                 <Users className="w-4 h-4" /> Justices & Panels
              </h4>
              <div className="flex flex-wrap gap-2">
                 {judges.map((judge, i) => (
                    <span key={`judge-${i}`} className="px-3 py-1.5 bg-gray-100 text-gray-800 border border-gray-200 rounded-md text-sm font-medium">
                       {judge}
                    </span>
                 ))}
              </div>
            </div>
        )}
      </div>
    </section>
  );
}
