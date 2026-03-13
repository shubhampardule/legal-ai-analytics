import { useEffect, useRef } from "react";
import Chart from "chart.js/auto";

const UI_FONT = "'IBM Plex Sans', 'Segoe UI', sans-serif";
const MONO_FONT = "'IBM Plex Mono', 'Consolas', monospace";

// Match exactly to the new index.css tokens
// Since chart.js needs raw hex/rgba values, we approximate the base theme tokens:
const COLORS = {
  text: "#111827",       // var(--color-text-primary)
  muted: "#6b7280",      // var(--color-text-secondary)
  border: "#e5e7eb",     // var(--color-border-subtle)
  grid: "#f3f4f6",       // var(--color-surface-elevated)
  accepted: "#166534",   // var(--color-status-accepted-text)
  acceptedBg: "#dcfce7", // var(--color-status-accepted-bg)
  rejected: "#991b1b",   // var(--color-status-rejected-text)
  rejectedBg: "#fee2e2", // var(--color-status-rejected-bg)
  primary: "#0369a1",    // a neutral strong tone
  primaryBg: "#f0f9ff",  // a neutral soft tone
};

function formatPercent(value, digits = 1) {
  if (!Number.isFinite(value)) return "--";
  return (value * 100).toFixed(digits) + "%";
}

function formatCompactNumber(value, digits = 2) {
  if (!Number.isFinite(value)) return "--";
  return Number(value).toFixed(digits);
}

function sumField(items, key) {
  return items.reduce((total, item) => total + Number(item?.[key] ?? 0), 0);
}

function StatItem({ label, value, hint }) {
  return (
    <div className="flex flex-col py-3 border-b border-[var(--color-border-subtle)] last:border-0">
      <span className="text-xs font-semibold uppercase tracking-widest text-[var(--color-text-tertiary)]">{label}</span>
      <span className="mt-1 font-serif text-2xl text-[var(--color-text-primary)]">{value}</span>
      {hint && <span className="mt-1 text-xs text-[var(--color-text-secondary)] leading-relaxed">{hint}</span>}
    </div>
  );
}

function ChartCanvas({ config }) {
  const canvasRef = useRef(null);
  const chartRef = useRef(null);

  useEffect(() => {
    if (!canvasRef.current || !config) return;

    chartRef.current?.destroy();
    chartRef.current = new Chart(canvasRef.current, config);

    return () => {
      chartRef.current?.destroy();
      chartRef.current = null;
    };
  }, [config]);

  return (
    <div className="relative w-full h-[180px] mt-4">
      <canvas ref={canvasRef} />
    </div>
  );
}

function buildTermConfig(explanation) {
  const acceptedTerms = explanation?.top_term_contributions?.accepted || [];
  const rejectedTerms = explanation?.top_term_contributions?.rejected || [];

  const allTerms = [
    ...acceptedTerms.map(t => ({ term: t.term, val: Number(t.contribution), type: 'accepted' })),
    ...rejectedTerms.map(t => ({ term: t.term, val: -Number(t.contribution), type: 'rejected' }))
  ];

  allTerms.sort((a, b) => Math.abs(b.val) - Math.abs(a.val));
  const topTerms = allTerms.slice(0, 5);

  return {
    type: "bar",
    data: {
      labels: topTerms.map(t => t.term),
      datasets: [{
        data: topTerms.map(t => t.val),
        backgroundColor: topTerms.map(t => t.type === 'accepted' ? COLORS.acceptedBg : COLORS.rejectedBg),
        borderColor: topTerms.map(t => t.type === 'accepted' ? COLORS.accepted : COLORS.rejected),
        borderWidth: 1,
        borderRadius: 4
      }]
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (c) => "Impact: " + Math.abs(c.raw).toFixed(3) + " (" + topTerms[c.dataIndex].type + ")"
          }
        }
      },
      scales: {
        x: {
          display: true,
          grid: { color: COLORS.border },
          ticks: { color: COLORS.muted, font: { family: UI_FONT, size: 10 } }
        },
        y: {
          grid: { display: false },
          ticks: { color: COLORS.text, font: { family: UI_FONT, size: 11, weight: '500' } }
        }
      }
    }
  };
}

function buildEvidenceConfig(supporting, opposing) {
  return {
    type: "bar",
    data: {
      labels: ["Supports", "Opposes"],
      datasets: [{
        data: [supporting, opposing],
        backgroundColor: [COLORS.acceptedBg, COLORS.rejectedBg],
        borderColor: [COLORS.accepted, COLORS.rejected],
        borderWidth: 1,
        borderRadius: 4
      }]
    },
    options: {
      indexAxis: "y",
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { display: false },
        y: { 
          grid: { display: false },
          ticks: { color: COLORS.text, font: { family: UI_FONT, size: 10 } }
        }
      }
    }
  };
}

export function AnalyticsChartsRow({ prediction, explanation }) {
  const acceptedProb = Number(prediction?.accepted_probability ?? 0);
  const rejectedProb = Number(prediction?.rejected_probability ?? 0);
  const predictedAccepted = prediction?.predicted_label === "accepted";

  const supporting = explanation?.sentence_evidence?.supporting ?? [];
  const opposing = explanation?.sentence_evidence?.opposing ?? [];
  const supportKey = predictedAccepted ? "accepted_evidence" : "rejected_evidence";
  const opposeKey = predictedAccepted ? "rejected_evidence" : "accepted_evidence";
  
  const supportScore = sumField(supporting, supportKey);
  const opposeScore = sumField(opposing, opposeKey);

  return (
    <div className="flex flex-col gap-6 pt-4">
      <div className="rounded-xl border border-[var(--color-border-subtle)] bg-[var(--color-surface-base)] p-4">
        <h4 className="text-sm font-semibold text-[var(--color-text-secondary)]">Top Term Impacts</h4>
        <ChartCanvas config={buildTermConfig(explanation)} />
      </div>
      
      <div className="rounded-xl border border-[var(--color-border-subtle)] bg-[var(--color-surface-base)] p-4">
        <h4 className="text-sm font-semibold text-[var(--color-text-secondary)]">Evidence Distribution</h4>
        <p className="text-xs text-[var(--color-text-tertiary)] mt-1">Weighted sum of passages</p>
        <ChartCanvas config={buildEvidenceConfig(supportScore, opposeScore)} />
      </div>
    </div>
  );
}

export function AnalyticsCardRow({ input, prediction, explanation, retrieval }) {
  const acceptedProb = Number(prediction?.accepted_probability ?? 0);
  const rejectedProb = Number(prediction?.rejected_probability ?? 0);
  const confidence = prediction?.predicted_label === "accepted" ? acceptedProb : rejectedProb;
  const strength = confidence >= 0.8 ? "Strong" : confidence >= 0.65 ? "Moderate" : "Weak";

  const evidenceCount = (explanation?.sentence_evidence?.supporting?.length || 0) + 
                        (explanation?.sentence_evidence?.opposing?.length || 0);

  const results = retrieval?.results ?? [];
  const avgSim = results.length ? results.reduce((sum, r) => sum + Number(r.similarity_score || 0), 0) / results.length : NaN;
  const matchQual = Number.isFinite(avgSim) ? (avgSim >= 0.85 ? "Good" : avgSim >= 0.7 ? "Moderate" : "Weak") : "Unknown";
  
  return (
    <div className="mt-6 flex flex-col gap-2 rounded-xl bg-[var(--color-surface-base)] border border-[var(--color-border-subtle)] p-5">
      <StatItem 
        label="Prediction Strength" 
        value={prediction?.predicted_label ? strength : "Pending"} 
        hint={prediction?.predicted_label ? "Outcome confidence: " + formatPercent(confidence) : null} 
      />
      <StatItem 
        label="Evidence Found" 
        value={evidenceCount || "0"} 
        hint={evidenceCount ? "Key passages extracted directly from text" : "Awaiting case analysis"} 
      />
      <StatItem 
        label="Case Match Quality" 
        value={matchQual} 
        hint={Number.isFinite(avgSim) ? "Average cosine similarity: " + formatPercent(avgSim) : "No references retrieved"} 
      />
      
      <div className="mt-2 pt-4 border-t border-[var(--color-border-subtle)]">
        <details className="group">
          <summary className="cursor-pointer text-xs font-semibold uppercase tracking-widest text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] transition-colors">
             System Information &rarr;
          </summary>
          <div className="mt-4 flex flex-col gap-2 text-xs font-mono text-[var(--color-text-tertiary)] bg-[var(--color-surface-elevated)] p-3 rounded-lg">
             <p>Input text length: {input?.clean_char_length || 0} chars</p>
             <p>Model bounded text: {explanation?.text_summary?.model_text_char_length || 0} chars</p>
             <p>Strategy: {input?.truncated_for_model ? "Truncated head/tail" : "Full ingestion"}</p>
          </div>
        </details>
      </div>
    </div>
  );
}
