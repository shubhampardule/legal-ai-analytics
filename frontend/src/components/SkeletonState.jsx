import { useEffect, useState } from "react";
import { Loader2 } from "lucide-react";

const steps = [
  "Processing text...",
  "Predicting outcome...",
  "Finding evidence...",
  "Searching similar cases...",
  "Finalizing analysis..."
];

export function SkeletonState() {
  const [stepIndex, setStepIndex] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setStepIndex((prev) => Math.min(prev + 1, steps.length - 1));
    }, 1500); // changes every 1.5s
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="flex flex-col gap-8 animate-in fade-in duration-500 w-full">
      <div className="flex items-center w-full mb-2">
         <div className="flex items-center gap-3 text-[var(--color-text-secondary)]">
           <Loader2 className="h-6 w-6 animate-spin" />
           <span className="font-serif text-2xl text-[var(--color-text-primary)]">{steps[stepIndex]}</span>
         </div>
      </div>
      
      {/* Verdict Dashboard Top Skeleton */}
      <div className="grid gap-6 2xl:grid-cols-3 w-full">
        <div className="2xl:col-span-2 flex flex-col gap-6">
          {/* PredictionHeroCard Skeleton */}
          <div className="h-[200px] w-full rounded-2xl bg-[var(--color-surface-elevated)] animate-pulse border border-[var(--color-border-subtle)]"></div>
          {/* AnalysisSummaryCard Skeleton */}
          <div className="h-[120px] w-full rounded-2xl bg-[var(--color-surface-elevated)] animate-pulse border border-[var(--color-border-subtle)]"></div>
        </div>
        <div className="2xl:col-span-1">
          {/* Technical Metrics Skeleton */}
          <div className="h-full min-h-[340px] w-full rounded-2xl bg-[var(--color-surface-elevated)] animate-pulse border border-[var(--color-border-subtle)]"></div>
        </div>
      </div>

      {/* Deep Analysis Evidence row Skeleton */}
      <div className="grid gap-6 2xl:grid-cols-3 w-full">
        <div className="2xl:col-span-2">
          {/* EvidenceSentencesCard Skeleton */}
          <div className="h-[300px] w-full rounded-2xl bg-[var(--color-surface-elevated)] animate-pulse border border-[var(--color-border-subtle)]"></div>
        </div>
        <div className="2xl:col-span-1">
          {/* KeyTermsCard Skeleton */}
          <div className="h-[300px] w-full rounded-2xl bg-[var(--color-surface-elevated)] animate-pulse border border-[var(--color-border-subtle)]"></div>
        </div>
      </div>
      
      {/* Similar Cases list Skeleton */}
      <div className="w-full">
         <div className="h-[400px] w-full rounded-2xl bg-[var(--color-surface-elevated)] animate-pulse border border-[var(--color-border-subtle)]"></div>
      </div>
    </div>
  );
}
