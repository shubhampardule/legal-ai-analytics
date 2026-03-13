import { Brain } from "lucide-react";

function getConfidenceLabel(score) {
  if (score >= 0.8) return "High";
  if (score >= 0.65) return "Moderate";
  return "Low";
}

export function PredictionHeroCard({ prediction }) {
  if (!prediction) return null;

  const accepted = prediction.accepted_probability ?? 0;
  const rejected = prediction.rejected_probability ?? 0;
  const isAccepted = prediction.predicted_label === "accepted";
  const confidenceScore = Math.max(accepted, rejected);
  const confidencePercent = Math.round(confidenceScore * 100);
  const confidenceLabel = getConfidenceLabel(confidenceScore);
  const verdictText = isAccepted ? "Likely to be Accepted" : "Likely to be Rejected";
  
  const statusColors = isAccepted
    ? "bg-[var(--color-status-accepted-bg)] border-[var(--color-status-accepted-border)] text-[var(--color-status-accepted-text)]"
    : "bg-[var(--color-status-rejected-bg)] border-[var(--color-status-rejected-border)] text-[var(--color-status-rejected-text)]";

  const meterColor = isAccepted ? "bg-[var(--color-status-accepted-text)]" : "bg-[var(--color-status-rejected-text)]";

  // Replaces the old panel-surface with our exact new tokens
  return (
    <section className={`relative overflow-hidden rounded-2xl border p-8 shadow-[var(--shadow-md-subtle)] ${statusColors}`}>
      
      {/* Background ambient gradient based on status - subtle */}
      <div className="pointer-events-none absolute inset-0 bg-gradient-to-br from-white/40 to-transparent"></div>
      
      <div className="relative z-10 flex flex-col gap-6 lg:flex-row lg:items-end lg:justify-between">
        <div className="flex-1">
          <div className="flex items-center gap-2">
            <Brain className="h-4 w-4" />
            <p className="text-xs font-semibold uppercase tracking-widest opacity-80">
              Predicted Outcome
            </p>
          </div>
          
          <h2 className="mt-4 font-serif text-5xl sm:text-6xl tracking-tight">
            {verdictText}
          </h2>
          <p className="mt-3 max-w-xl text-lg leading-relaxed opacity-90 font-medium">
            This case is {verdictText.toLowerCase()} with {confidenceLabel.toLowerCase()} confidence ({confidencePercent}%).
          </p>
        </div>

        {/* Technical abstract - explicitly boxed off with strong border radius matching scale */}
        <div className="shrink-0 w-full lg:w-72 rounded-xl bg-white/60 p-5 shadow-sm backdrop-blur-sm border border-white/40">
          <div className="flex justify-between items-end mb-2">
             <p className="text-xs font-semibold uppercase tracking-widest text-[var(--color-text-secondary)]">
              Confidence
            </p>
            <span className="font-mono text-sm font-bold">{confidencePercent}%</span>
          </div>
         
          <div className="h-3 w-full overflow-hidden rounded-full bg-black/5 mt-3 shadow-inner">
            <div
              className={`h-full rounded-full transition-all duration-1000 ease-out ${meterColor}`}
              style={{ width: `${Math.max(5, confidencePercent)}%` }}
            />
          </div>
        </div>
      </div>
      
    </section>
  );
}
