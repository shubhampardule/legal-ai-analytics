import { Tags } from "lucide-react";

function TermGroup({ title, toneClass, indicatorClass, items }) {
  const maxContribution = items?.length
    ? Math.max(...items.map((item) => Number(item?.contribution ?? 0)))
    : 0;
    
  if (!items?.length) return null;

  return (
    <div className="mt-8 first:mt-6">
      <h4 className="text-xs font-semibold uppercase tracking-widest text-[var(--color-text-tertiary)] mb-4">
        {title}
      </h4>
      <div className="flex flex-wrap gap-2.5">
        {items.map((item) => {
           const score = Number(item?.contribution ?? 0);
           const relativeScore = maxContribution ? score / maxContribution : 0;
           
           // Scale font size slightly, but not as extreme as before to keep baseline grid
           const fontSize = 0.85 + (relativeScore * 0.15);
           const opacity = 0.6 + (relativeScore * 0.4);
           
           return (
             <div 
               key={`${title}-${item.term}`}
               className={`relative flex items-center justify-center rounded-lg border px-3 py-1.5 transition-all hover:scale-105 ${toneClass}`}
               style={{ 
                  fontSize: `${fontSize}rem`,
               }}
             >
                <div 
                   className="absolute inset-0 rounded-lg opacity-10 pointer-events-none"
                   style={{ backgroundColor: `currentcolor` }}
                />
                
                <span className="relative z-10 flex items-center gap-2" style={{ opacity }}>
                  <span className="font-medium text-[var(--color-text-primary)]">{item.term}</span>
                </span>
             </div>
           );
        })}
      </div>
    </div>
  );
}

export function KeyTermsCard({ explanation }) {
  const accepted = explanation?.top_term_contributions?.accepted ?? [];
  const rejected = explanation?.top_term_contributions?.rejected ?? [];
  const hasTerms = accepted.length > 0 || rejected.length > 0;

  if (!hasTerms) return null;

  return (
    <section className="rounded-2xl border border-[var(--color-border-subtle)] bg-[var(--color-surface-panel)] p-8 shadow-[var(--shadow-sm-subtle)]">
      <div className="flex items-center gap-2 mb-2">
        <Tags className="h-4 w-4 text-[var(--color-text-tertiary)]" />
        <p className="text-xs font-semibold uppercase tracking-widest text-[var(--color-text-secondary)]">
          Vocabulary Impact
        </p>
      </div>
      
      <h3 className="font-serif text-2xl text-[var(--color-text-primary)]">
        Key Influencing Words
      </h3>
      
      <TermGroup
        title="Drove Acceptance"
        toneClass="border-[var(--color-status-accepted-border)]/50 bg-[var(--color-status-accepted-bg)]/30"
        indicatorClass="bg-[var(--color-status-accepted-text)]"
        items={accepted}
      />
      
      <TermGroup
        title="Drove Rejection"
        toneClass="border-[var(--color-status-rejected-border)]/50 bg-[var(--color-status-rejected-bg)]/30"
        indicatorClass="bg-[var(--color-status-rejected-text)]"
        items={rejected}
      />
    </section>
  );
}
