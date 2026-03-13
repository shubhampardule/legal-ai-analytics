import { startTransition, useEffect, useRef, useState } from "react";
import { analyzeCase, fetchHealth, fetchMeta } from "./api/client";
import { AnalysisSummaryCard } from "./components/AnalysisSummaryCard";
import { AnalyticsCardRow, AnalyticsChartsRow } from "./components/AnalyticsCardRow";
import { EvidenceSentencesCard } from "./components/EvidenceSentencesCard";
import { HeaderBar } from "./components/HeaderBar";
import { CaseExplorerModal } from "./components/CaseExplorerModal";
import { CaseDetailsModal } from "./components/CaseDetailsModal";
import { InputPanel } from "./components/InputPanel";
import { KeyTermsCard } from "./components/KeyTermsCard";
import { PredictionHeroCard } from "./components/PredictionHeroCard";
import { SimilarCasesList } from "./components/SimilarCasesList";
import { EntityExtractionCard } from "./components/EntityExtractionCard";
import { CaseChatCard } from "./components/CaseChatCard";
import { Footer } from "./components/Footer";
import { Layers, Scale } from "lucide-react";

const SAMPLE_CASES = ["1980_211", "1973_261", "1983_326"];

function EmptyState() {
  return (
    <div className="flex h-full w-full flex-col items-center justify-center rounded-2xl border border-dashed border-[var(--color-border-strong)] bg-[var(--color-surface-base)]/50 p-6 sm:p-12 text-center">
      <div className="flex h-20 w-20 items-center justify-center rounded-full bg-[var(--color-surface-elevated)] text-[var(--color-text-tertiary)] shadow-inner mb-6">
        <Scale className="h-8 w-8" />
      </div>
      <h2 className="font-serif text-2xl text-[var(--color-text-primary)]">
        Legal Copilot Ready
      </h2>
      <p className="mt-2 max-w-md text-sm leading-relaxed text-[var(--color-text-secondary)]">
        Enter a case ID or paste the judgment text in the sidebar. The system will predict the outcome, extract key evidence, and find relevant historical precedents.
      </p>
    </div>
  );
}

export default function App() {
  const [apiReady, setApiReady] = useState(null);
  const [apiMeta, setApiMeta] = useState(null);
  const [inputMode, setInputMode] = useState("caseId"); // Default to caseId for quick samples
  const [textValue, setTextValue] = useState("");
  const [caseIdValue, setCaseIdValue] = useState("1980_211");
  const [isProcessing, setIsProcessing] = useState(false);
  const [errorMsg, setErrorMsg] = useState(null);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [similarityFilters, setSimilarityFilters] = useState({
    outcome: null,
    year_from: null,
    year_to: null,
  });
  const [history, setHistory] = useState(() => {
    const saved = localStorage.getItem("legal_ai_history");
    return saved ? JSON.parse(saved) : [];
  });
  const [showExplorer, setShowExplorer] = useState(false);
    const [showSourceDocument, setShowSourceDocument] = useState(false);
  const resultsContentRef = useRef(null);

  useEffect(() => {
    async function checkBackend() {
      try {
        const [healthRes, metaRes] = await Promise.all([
          fetchHealth().catch(() => ({ status: "error" })),
          fetchMeta().catch(() => ({})),
        ]);
        startTransition(() => {
          setApiReady(healthRes.status === "ok");
          setApiMeta(metaRes?.data || metaRes);
        });
      } catch (err) {
        startTransition(() => setApiReady(false));
      }
    }
    checkBackend();
  }, []);

  useEffect(() => {
    if (analysisResult && !isProcessing && window.innerWidth < 1024) {
      setTimeout(() => {
        resultsContentRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
      }, 100);
    }
  }, [analysisResult, isProcessing]);

  async function handleAnalyze(filterOverride = null) {
    setErrorMsg(null);
    setIsProcessing(true);

    try {
      const payload = { top_k: 24 };
      const effectiveFilters = filterOverride ?? similarityFilters;
      if (effectiveFilters.outcome) payload.outcome = effectiveFilters.outcome;
      if (Number.isInteger(effectiveFilters.year_from)) payload.year_from = effectiveFilters.year_from;
      if (Number.isInteger(effectiveFilters.year_to)) payload.year_to = effectiveFilters.year_to;
      if (inputMode === "text") {
        if (!textValue.trim()) throw new Error("Please enter case text.");
        payload.text = textValue.trim();
      } else {
        if (!caseIdValue.trim()) throw new Error("Please enter a case ID.");
        payload.case_id = caseIdValue.trim();
      }

      const startTime = performance.now();
      const response = await analyzeCase(payload);
      const endTime = performance.now();

      startTransition(() => {
        const resultData = { ...response.data, _timings: { total_ms: endTime - startTime } };
        setAnalysisResult(resultData);
        
        // Save to history
        setHistory(prev => {
          const entry = {
            id: Date.now().toString(),
            type: inputMode,
            query: inputMode === "caseId" ? caseIdValue.trim() : textValue.substring(0, 40) + "...",
            verdict: resultData.prediction?.predicted_label || "unknown",
            timestamp: new Date().toISOString(),
            fullResult: resultData
          };
          // Keep last 10 entries, remove duplicates if caseId
          const filtered = prev.filter(item => !(item.type === "caseId" && inputMode === "caseId" && item.query === caseIdValue.trim()));
          const newHistory = [entry, ...filtered].slice(0, 10);
          localStorage.setItem("legal_ai_history", JSON.stringify(newHistory));
          return newHistory;
        });
      });
    } catch (err) {
      startTransition(() => {
        setErrorMsg(err.message || "Failed to analyze case. The backend might be unreachable.");
        setAnalysisResult(null);
      });
    } finally {
      startTransition(() => {
        setIsProcessing(false);
      });
    }
  }

  function handleSimilarityFilterChange(field, value) {
    setSimilarityFilters((prev) => ({ ...prev, [field]: value }));
  }

  function handleSimilarityFilterReset() {
    setSimilarityFilters({ outcome: null, year_from: null, year_to: null });
  }

  function handleClear() {
    startTransition(() => {
      setTextValue("");
      setCaseIdValue("");
      setErrorMsg(null);
      setAnalysisResult(null);
    });
  }


  function handleClearHistory() {
    setHistory([]);
    localStorage.removeItem('legal_ai_history');
  }

  function handleRestoreHistory(fullResult, type, query) {
    startTransition(() => {
      setInputMode(type);
      if (type === 'text') {
        setTextValue(query);
      } else {
        setCaseIdValue(query);
      }
      setAnalysisResult(fullResult);
      setErrorMsg(null);
    });
  }

  return (
    <div className="flex min-h-[100dvh] lg:h-screen w-full flex-col bg-[var(--color-surface-base)] font-sans text-[var(--color-text-primary)] antialiased lg:overflow-hidden print:h-auto print:overflow-visible">
<div className="z-20 shrink-0 bg-[var(--color-surface-panel)] border-b border-[var(--color-border-subtle)] shadow-[var(--shadow-sm-subtle)] px-6 py-4 print:hidden">
        <HeaderBar
          apiHealthy={apiReady}
          apiMeta={apiMeta}
          onBrowseDatabase={() => setShowExplorer(true)}
        />
      </div>

      {showExplorer && (
        <CaseExplorerModal onClose={() => setShowExplorer(false)} />
      )}

      {showSourceDocument && (
        <CaseDetailsModal 
          caseId={inputMode === "caseId" ? caseIdValue : null}
          rawText={inputMode === "text" ? textValue : null}
          highlights={{
            supporting: analysisResult?.explanation?.sentence_evidence?.supporting?.map(s => s.text) || [],
            opposing: analysisResult?.explanation?.sentence_evidence?.opposing?.map(s => s.text) || []
          }}
          onClose={() => setShowSourceDocument(false)} 
        />
      )}

      {/* Main App Layout */}
      <div className="flex flex-1 flex-col lg:flex-row lg:overflow-hidden relative print:block print:overflow-visible">
        
        {/* LEFT SIDEBAR: Controls & Input */}
        <aside className="w-full lg:w-[420px] xl:w-[460px] shrink-0 border-b lg:border-b-0 lg:border-r border-[var(--color-border-subtle)] bg-[var(--color-surface-panel)] lg:overflow-y-auto z-10 flex flex-col print:hidden">
          <div className="p-6">
             <InputPanel history={history} onRestoreHistory={handleRestoreHistory} onClearHistory={handleClearHistory}
                inputMode={inputMode}
                setInputMode={setInputMode}
                textValue={textValue}
                setTextValue={setTextValue}
                caseIdValue={caseIdValue}
                setCaseIdValue={setCaseIdValue}
                isLoading={isProcessing}
                errorMessage={errorMsg}
                sampleCases={SAMPLE_CASES}
                onAnalyze={handleAnalyze}
                onClear={handleClear}
                onSampleSelect={(id) => {
                  setInputMode("caseId");
                  setCaseIdValue(id);
                }}
              />
          </div>
        </aside>

        {/* RIGHT AREA: Results / Canvas */}
        <main className="flex-1 relative lg:overflow-y-auto bg-[var(--color-surface-base)] p-4 sm:p-6 lg:p-10 print:block print:overflow-visible print:p-0 print:bg-transparent" ref={resultsContentRef}>
          
          <div className="mx-auto max-w-6xl h-full print:h-auto pb-20 print:pb-0">
            {isProcessing ? (
              <div className="flex h-full w-full flex-col items-center justify-center">
                  <div className="relative mb-8 h-2 w-64 overflow-hidden rounded-full bg-[var(--color-border-subtle)]">
                    <div className="loading-bar absolute inset-y-0 left-0 w-1/2 rounded-full bg-[var(--color-text-primary)]"></div>
                  </div>
                  <p className="font-serif text-2xl text-[var(--color-text-primary)] animate-pulse">
                    Analyzing precedents & evidence...
                  </p>
                  <p className="mt-3 max-w-md text-center text-sm text-[var(--color-text-secondary)]">
                    Extracting embeddings, running similarity search against knowledge base, and synthesizing explainable predictions.
                  </p>
              </div>
            ) : analysisResult ? (
              <div className="flex flex-col gap-8 animate-in fade-in slide-in-from-bottom-4 duration-700">
                 {/* Verdict Dashboard Top */}
                 <div className="grid gap-6 2xl:grid-cols-3">
                    <div className="2xl:col-span-2 flex flex-col gap-6">
                       <PredictionHeroCard prediction={analysisResult.prediction} />
                       <AnalysisSummaryCard
                          summary={analysisResult.summary}
                          prediction={analysisResult.prediction}
                          explanation={analysisResult.explanation}
                        />
                    </div>
                    <div className="2xl:col-span-1">
                        <section className="h-full rounded-2xl border border-[var(--color-border-subtle)] bg-[var(--color-surface-panel)] p-6 shadow-[var(--shadow-sm-subtle)] flex flex-col">
                          <p className="mb-4 text-xs font-semibold uppercase tracking-widest text-[var(--color-text-tertiary)]">
                            Technical Metrics
                          </p>
                          <div className="flex-1 min-h-0 min-w-0 space-y-6 overflow-hidden">
                            <AnalyticsChartsRow {...analysisResult} />
                            <AnalyticsCardRow {...analysisResult} />
                          </div>
                        </section>
                    </div>
                 </div>

                 {/* Deep Analysis Evidence row */}
                 <div className="grid gap-6 2xl:grid-cols-3">
                    <div className="2xl:col-span-2">
                       <EvidenceSentencesCard explanation={analysisResult.explanation} onViewDocument={() => setShowSourceDocument(true)} />
                    </div>
                    <div className="2xl:col-span-1">
                       <KeyTermsCard explanation={analysisResult.explanation} />
                         <div className="mt-6"><EntityExtractionCard entities={analysisResult.entities} /></div>
                    </div>
                 </div>

                 {/* Bottom Similar Cases */}
                 <div className="w-full">
                    <div className="mb-6">
                      <CaseChatCard
                        inputMode={inputMode}
                        caseIdValue={caseIdValue}
                        textValue={textValue}
                        disabled={!analysisResult}
                      />
                    </div>
                    <SimilarCasesList
                      retrieval={analysisResult.retrieval}
                      filters={similarityFilters}
                      onFilterChange={handleSimilarityFilterChange}
                      onApplyFilters={() => handleAnalyze()}
                      onResetFilters={() => {
                        const resetFilters = { outcome: null, year_from: null, year_to: null };
                        setSimilarityFilters(resetFilters);
                        handleAnalyze(resetFilters);
                      }}
                    />
                 </div>
              </div>
            ) : !errorMsg ? (
               <EmptyState />
            ) : null}
            
            {/* Global Footer */}
            <Footer />
          </div>
        </main>
      </div>
    </div>
  );
}
















