import { startTransition, useEffect, useRef, useState } from "react";
import { analyzeCase, fetchHealth, fetchMeta } from "./api/client";
import { AnalyticsCardRow } from "./components/AnalyticsCardRow";
import { EvidenceSentencesCard } from "./components/EvidenceSentencesCard";
import { HeaderBar } from "./components/HeaderBar";
import { InputPanel } from "./components/InputPanel";
import { KeyTermsCard } from "./components/KeyTermsCard";
import { PredictionHeroCard } from "./components/PredictionHeroCard";
import { SimilarCasesList } from "./components/SimilarCasesList";

const SAMPLE_CASES = ["1980_211", "1973_261", "1983_326"];

function EmptyState() {
  return (
    <section className="panel-surface rounded-[2rem] border border-[var(--border-soft)] p-8 shadow-[0_24px_80px_rgba(0,0,0,0.28)]">
      <p className="text-xs uppercase tracking-[0.38em] text-[var(--accent-gold)]">
        Ready For Analysis
      </p>
      <h2 className="mt-4 font-title text-4xl text-[var(--text-main)]">
        Run a legal case analysis
      </h2>
      <p className="mt-4 max-w-2xl text-sm leading-7 text-[var(--text-muted)]">
        Submit a dataset case ID or paste cleaned case text to predict the likely
        outcome, inspect the strongest evidence, and retrieve similar cases for
        research.
      </p>
      <div className="mt-8 grid gap-4 md:grid-cols-3">
        {[
          "Predict likely outcome",
          "Inspect evidence terms and sentences",
          "Retrieve similar cases for research",
        ].map((item) => (
          <div
            key={item}
            className="rounded-[1.5rem] border border-[var(--border-soft)] bg-[rgba(255,255,255,0.02)] p-5"
          >
            <p className="font-ui text-sm text-[var(--text-main)]">{item}</p>
          </div>
        ))}
      </div>
    </section>
  );
}

function Banner({ title, message, tone = "warning" }) {
  const toneClass =
    tone === "error"
      ? "border-[rgba(182,75,60,0.55)] bg-[rgba(182,75,60,0.12)] text-[#ffd5cc]"
      : "border-[rgba(200,155,60,0.45)] bg-[rgba(200,155,60,0.12)] text-[#b38025]";

  return (
    <div className={`rounded-[1.35rem] border px-5 py-4 ${toneClass}`}>
      <p className="font-ui text-xs uppercase tracking-[0.32em]">{title}</p>
      <p className="mt-2 font-ui text-sm leading-6">{message}</p>
    </div>
  );
}

export default function App() {
  const [inputMode, setInputMode] = useState("text");
  const [textValue, setTextValue] = useState("");
  const [caseIdValue, setCaseIdValue] = useState("");
  const [apiHealthy, setApiHealthy] = useState(null);
  const [apiMeta, setApiMeta] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [warnings, setWarnings] = useState([]);
  const [errorMessage, setErrorMessage] = useState("");
  const [responseData, setResponseData] = useState(null);
  const resultsRef = useRef(null);

  useEffect(() => {
    let active = true;

    async function loadApiState() {
      try {
        const [healthResponse, metaResponse] = await Promise.all([
          fetchHealth(),
          fetchMeta(),
        ]);
        if (!active) {
          return;
        }
        startTransition(() => {
          setApiHealthy(
            Boolean(
              healthResponse.data.prediction_model_loaded &&
                healthResponse.data.similarity_index_loaded &&
                healthResponse.data.similarity_encoder_loaded,
            ),
          );
          setApiMeta(metaResponse.data);
        });
      } catch {
        if (!active) {
          return;
        }
        startTransition(() => {
          setApiHealthy(false);
          setApiMeta(null);
        });
      }
    }

    loadApiState();
    return () => {
      active = false;
    };
  }, []);

  async function handleAnalyze() {
    const activeValue = inputMode === "text" ? textValue.trim() : caseIdValue.trim();
    if (!activeValue) {
      setErrorMessage(
        inputMode === "text"
          ? "Paste some case text before running the analysis."
          : "Enter a case ID before running the analysis.",
      );
      return;
    }

    setIsLoading(true);
    setErrorMessage("");
    setWarnings([]);

    const payload = {
      include_explanation: true,
      include_similar_cases: true,
      top_k: 5,
    };

    if (inputMode === "text") {
      payload.text = textValue;
    } else {
      payload.case_id = caseIdValue.trim();
    }

    try {
      const envelope = await analyzeCase(payload);
      startTransition(() => {
        setResponseData(envelope.data);
        setWarnings(envelope.warnings || []);
      });
      window.setTimeout(() => {
        resultsRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
      }, 120);
    } catch (error) {
      setResponseData(null);
      setErrorMessage(error.message);
    } finally {
      setIsLoading(false);
    }
  }

  function handleClear() {
    setTextValue("");
    setCaseIdValue("");
    setWarnings([]);
    setErrorMessage("");
    setResponseData(null);
  }

  function handleSampleSelect(caseId) {
    setInputMode("caseId");
    setCaseIdValue(caseId);
    setErrorMessage("");
  }

  const prediction = responseData?.prediction ?? null;
  const explanation = responseData?.explanation ?? null;
  const retrieval = responseData?.retrieval ?? null;
  const inputInfo = responseData?.input ?? null;

  return (
    <div className="relative min-h-screen overflow-x-hidden bg-[var(--bg-ink)] text-[var(--text-main)]">
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_top,_rgba(200,155,60,0.18),_transparent_38%),linear-gradient(180deg,_rgba(255,255,255,0.02),_transparent)]" />
      <div className="absolute inset-0 opacity-30 [background-image:linear-gradient(rgba(255,255,255,0.035)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.035)_1px,transparent_1px)] [background-size:72px_72px]" />

      <main className="relative z-10 mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-8 lg:py-10">
        <HeaderBar apiHealthy={apiHealthy} apiMeta={apiMeta} />

        <section className="mt-8 grid gap-6 lg:grid-cols-[minmax(0,0.92fr)_minmax(0,1.48fr)]">
          <div className="lg:sticky lg:top-6 lg:self-start min-w-0">
            <InputPanel
              inputMode={inputMode}
              setInputMode={setInputMode}
              textValue={textValue}
              setTextValue={setTextValue}
              caseIdValue={caseIdValue}
              setCaseIdValue={setCaseIdValue}
              isLoading={isLoading}
              errorMessage={errorMessage}
              sampleCases={SAMPLE_CASES}
              onAnalyze={handleAnalyze}
              onClear={handleClear}
              onSampleSelect={handleSampleSelect}
            />
          </div>

          <div ref={resultsRef} className="space-y-6 min-w-0">
            {warnings.length > 0 ? (
              <div className="space-y-3">
                {warnings.map((warning) => (
                  <Banner
                    key={warning}
                    title="Processing Note"
                    message={warning}
                    tone="warning"
                  />
                ))}
              </div>
            ) : null}

            {!responseData && !isLoading ? <EmptyState /> : null}

            {isLoading ? (
              <section className="panel-surface rounded-[2rem] border border-[var(--border-soft)] p-8 shadow-[0_24px_80px_rgba(0,0,0,0.28)]">
                <p className="text-xs uppercase tracking-[0.38em] text-[var(--accent-gold)]">
                  Working
                </p>
                <h2 className="mt-4 font-title text-3xl text-[var(--text-main)]">
                  Analyzing case text, generating explanation, and searching similar
                  cases...
                </h2>
                <div className="mt-6 h-2 overflow-hidden rounded-full bg-[rgba(0,0,0,0.06)]">
                  <div className="loading-bar h-full w-1/3 rounded-full bg-[var(--accent-gold)]" />
                </div>
              </section>
            ) : null}

            {responseData ? (
              <>
                <section className="panel-surface rounded-[2rem] border border-[var(--border-soft)] p-6 shadow-[0_24px_80px_rgba(0,0,0,0.28)]">
                  <div className="flex flex-wrap items-center gap-3">
                    <span className="rounded-full border border-[var(--border-soft)] px-3 py-1 text-xs uppercase tracking-[0.28em] text-[var(--accent-gold)]">
                      {inputInfo?.source === "case_id" ? "Dataset Case" : "Pasted Text"}
                    </span>
                    {inputInfo?.split ? (
                      <span className="rounded-full bg-[rgba(0,0,0,0.04)] px-3 py-1 text-xs text-[var(--text-muted)]">
                        Split: {inputInfo.split}
                      </span>
                    ) : null}
                    {inputInfo?.true_label ? (
                      <span className="rounded-full bg-[rgba(0,0,0,0.04)] px-3 py-1 text-xs text-[var(--text-muted)]">
                        True label: {inputInfo.true_label}
                      </span>
                    ) : null}
                    <span className="rounded-full bg-[rgba(0,0,0,0.04)] px-3 py-1 text-xs text-[var(--text-muted)]">
                      Clean chars: {inputInfo?.clean_char_length?.toLocaleString?.() ?? "-"}
                    </span>
                    {inputInfo?.truncated_for_model ? (
                      <span className="rounded-full border border-[rgba(179,128,37,0.3)] px-3 py-1 text-xs text-[#b38025]">
                        Truncated for model
                      </span>
                    ) : null}
                  </div>
                </section>

                <PredictionHeroCard prediction={prediction} />
                <AnalyticsCardRow
                  input={inputInfo}
                  prediction={prediction}
                  explanation={explanation}
                  retrieval={retrieval}
                />

                <section className="grid gap-6 xl:grid-cols-2">
                  <KeyTermsCard explanation={explanation} />
                  <EvidenceSentencesCard explanation={explanation} />
                </section>

                <SimilarCasesList retrieval={retrieval} />
              </>
            ) : null}
          </div>
        </section>
      </main>
    </div>
  );
}

