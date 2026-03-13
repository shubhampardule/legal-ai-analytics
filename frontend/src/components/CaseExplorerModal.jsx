import { useState, useEffect } from "react";
import { X, Database, ChevronLeft, ChevronRight, Loader2, FileText, ExternalLink, Search } from "lucide-react";
import { listCases } from "../api/client";
import { CaseDetailsModal } from "./CaseDetailsModal";

export function CaseExplorerModal({ onClose }) {
  const [cases, setCases] = useState([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const [page, setPage] = useState(1);
  const limit = 20;

  const [searchQuery, setSearchQuery] = useState("");
  const [submittedQuery, setSubmittedQuery] = useState("");
  const [outcome, setOutcome] = useState("");
  const [yearFrom, setYearFrom] = useState("");
  const [yearTo, setYearTo] = useState("");

  const [activeCaseId, setActiveCaseId] = useState(null);

  useEffect(() => {
    async function fetchPage() {
      try {
        setLoading(true);
        const offset = (page - 1) * limit;
        const res = await listCases(limit, offset, submittedQuery, {
          outcome: outcome || null,
          year_from: yearFrom ? Number(yearFrom) : null,
          year_to: yearTo ? Number(yearTo) : null,
        });
        setCases(res.data.items);
        setTotal(res.data.total);
        setError(null);
      } catch (err) {
        setError(err.message || "Failed to load cases.");
      } finally {
        setLoading(false);
      }
    }
    fetchPage();
  }, [page, submittedQuery, outcome, yearFrom, yearTo]);

  const handleSearch = (e) => {
    e.preventDefault();
    setPage(1);
    setSubmittedQuery(searchQuery);
  };
  const totalPages = Math.ceil(total / limit);

  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm animate-in fade-in duration-200">
      <div 
        className="relative flex flex-col w-full max-w-6xl h-[90vh] bg-[var(--color-surface-base)] rounded-2xl shadow-2xl border border-[var(--color-border-strong)] overflow-hidden"
        role="dialog"
      >
        {/* Header */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between px-6 py-4 border-b border-[var(--color-border-subtle)] bg-[var(--color-surface-panel)] gap-4">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-[var(--color-surface-elevated)] rounded-lg border border-[var(--color-border-subtle)]">
              <Database className="w-5 h-5 text-[var(--color-text-primary)]" />
            </div>
            <div>
                <h2 className="text-xl font-semibold font-serif text-[var(--color-text-primary)]">
                    Database Explorer
                </h2>
                <p className="text-xs text-[var(--color-text-secondary)] font-medium">Browsing {total.toLocaleString()} historic cases</p>
            </div>
          </div>
          
          <div className="flex items-center gap-4">
            <form onSubmit={handleSearch} className="relative flex items-center">
              <Search className="absolute left-3 w-4 h-4 text-[var(--color-text-tertiary)]" />
              <input
                type="text"
                placeholder="Search case ID or status..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-9 pr-4 py-2 w-64 text-sm bg-[var(--color-surface-base)] border border-[var(--color-border-strong)] rounded-lg focus:outline-none focus:ring-2 focus:ring-[var(--color-text-primary)] transition-shadow text-[var(--color-text-primary)] placeholder-[var(--color-text-tertiary)]"
              />
            </form>
            <select
              value={outcome}
              onChange={(e) => {
                setPage(1);
                setOutcome(e.target.value);
              }}
              className="px-3 py-2 text-sm bg-[var(--color-surface-base)] border border-[var(--color-border-strong)] rounded-lg"
            >
              <option value="">All outcomes</option>
              <option value="accepted">Accepted</option>
              <option value="rejected">Rejected</option>
            </select>
            <input
              type="number"
              placeholder="Year from"
              value={yearFrom}
              onChange={(e) => {
                setPage(1);
                setYearFrom(e.target.value);
              }}
              className="px-3 py-2 w-28 text-sm bg-[var(--color-surface-base)] border border-[var(--color-border-strong)] rounded-lg"
              min={1800}
              max={2100}
            />
            <input
              type="number"
              placeholder="Year to"
              value={yearTo}
              onChange={(e) => {
                setPage(1);
                setYearTo(e.target.value);
              }}
              className="px-3 py-2 w-24 text-sm bg-[var(--color-surface-base)] border border-[var(--color-border-strong)] rounded-lg"
              min={1800}
              max={2100}
            />
            <button
              onClick={onClose}
              className="p-2 rounded-full hover:bg-[var(--color-surface-elevated)] transition-colors text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)]"
              aria-label="Close dialog"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6 bg-[var(--color-surface-base)]">
          {loading && cases.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full gap-4 text-[var(--color-text-secondary)]">
              <Loader2 className="w-8 h-8 animate-spin" />
              <p>Loading database...</p>
            </div>
          ) : error && cases.length === 0 ? (
            <div className="flex items-center justify-center h-full text-[var(--color-status-rejected-text)] p-4 bg-[var(--color-status-rejected-bg)] rounded-xl border border-[var(--color-status-rejected-border)]">
              <p>{error}</p>
            </div>
          ) : (
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
              {cases.map((c) => {
                const isAccepted = c.label_name === "accepted";
                return (
                  <article 
                    key={c.id} 
                    className="flex flex-col p-5 bg-[var(--color-surface-panel)] border border-[var(--color-border-subtle)] rounded-xl hover:shadow-md hover:border-[var(--color-border-strong)] transition-all cursor-pointer group"
                    onClick={() => setActiveCaseId(c.id)}
                  >
                    <div className="flex justify-between items-start mb-3">
                        <div className="flex items-center gap-2">
                            <FileText className="w-4 h-4 text-[var(--color-text-tertiary)]" />
                            <span className="font-semibold text-sm text-[var(--color-text-primary)]">#{c.id}</span>
                        </div>
                        <span className={`text-[0.65rem] font-bold uppercase tracking-wider px-2 py-0.5 rounded-full ${
                            isAccepted 
                                ? "bg-[var(--color-status-accepted-bg)] text-[var(--color-status-accepted-text)]" 
                                : "bg-[var(--color-status-rejected-bg)] text-[var(--color-status-rejected-text)]"
                        }`}>
                            {isAccepted ? "Accepted" : "Rejected"}
                        </span>
                    </div>

                    <div className="mt-auto pt-4 flex items-center justify-between border-t border-[var(--color-border-subtle)]">
                        <span className="text-xs text-[var(--color-text-tertiary)] font-mono">
                            {c.clean_char_length.toLocaleString()} chars
                        </span>
                        <div className="flex items-center gap-1 text-[var(--color-text-secondary)] group-hover:text-[var(--color-text-primary)] transition-colors">
                            <span className="text-xs font-medium">View</span>
                            <ExternalLink className="w-3 h-3" />
                        </div>
                    </div>
                  </article>
                );
              })}
            </div>
          )}
        </div>
        
        {/* Pagination Footer */}
        <div className="flex items-center justify-between border-t border-[var(--color-border-subtle)] bg-[var(--color-surface-panel)] px-6 py-4">
            <span className="text-sm text-[var(--color-text-secondary)] font-medium">
                Showing {cases.length ? (page - 1) * limit + 1 : 0} to {Math.min(page * limit, total)} of {total.toLocaleString()} entries
            </span>
            <div className="flex items-center gap-2">
                <button
                    onClick={() => setPage(p => Math.max(1, p - 1))}
                    disabled={page === 1 || loading}
                    className="p-2 flex items-center gap-1 rounded-lg border border-[var(--color-border-subtle)] bg-[var(--color-surface-base)] text-[var(--color-text-primary)] hover:bg-[var(--color-surface-elevated)] disabled:opacity-50 disabled:pointer-events-none transition-colors"
                >
                    <ChevronLeft className="w-4 h-4" />
                </button>
                <div className="px-4 py-2 border border-[var(--color-border-subtle)] bg-[var(--color-surface-base)] rounded-lg text-sm font-semibold text-[var(--color-text-primary)]">
                    Page {page} of {totalPages || 1}
                </div>
                <button
                    onClick={() => setPage(p => Math.min(totalPages, p + 1))}
                    disabled={page === totalPages || loading}
                    className="p-2 flex items-center gap-1 rounded-lg border border-[var(--color-border-subtle)] bg-[var(--color-surface-base)] text-[var(--color-text-primary)] hover:bg-[var(--color-surface-elevated)] disabled:opacity-50 disabled:pointer-events-none transition-colors"
                >
                    <ChevronRight className="w-4 h-4" />
                </button>
            </div>
        </div>
      </div>

      {/* Nested Case Details Modal */}
      {activeCaseId && (
        <CaseDetailsModal 
          caseId={activeCaseId} 
          onClose={(e) => {
              if (e) e.stopPropagation();
              setActiveCaseId(null);
          }} 
        />
      )}
    </div>
  );
}
