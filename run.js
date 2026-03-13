const fs = require('fs');
let code = fs.readFileSync('frontend/src/components/InputPanel.jsx', 'utf8');

code = code.replace(/import \{ ([^}]+) \} from "lucide-react";/, 'import { useState } from "react";\nimport { $1, ChevronDown, ChevronUp } from "lucide-react";');

code = code.replace('export function InputPanel({ history, onRestoreHistory,', 'export function InputPanel({ history, onRestoreHistory, onClearHistory,');

const stateInjection = `const isTextMode = inputMode === "text";\n  const [showAllHistory, setShowAllHistory] = useState(false);\n  const displayedHistory = history ? (showAllHistory ? history : history.slice(0, 3)) : [];`;

code = code.replace('const isTextMode = inputMode === "text";', stateInjection);

const oldHistoryHeader = `<p className="text-xs font-semibold uppercase tracking-widest text-[var(--color-text-secondary)] mb-3 px-1">
                Recent Analyses
              </p>`;
const newHistoryHeader = `<div className="flex items-center justify-between mb-3 px-1">
                <p className="text-xs font-semibold uppercase tracking-widest text-[var(--color-text-secondary)]">
                  Recent Analyses
                </p>
                <button 
                  onClick={onClearHistory}
                  className="text-[10px] uppercase font-bold text-[var(--color-text-tertiary)] hover:text-red-500 transition-colors"
                  title="Clear all history"
                >
                  Clear History
                </button>
              </div>`;

code = code.replace(oldHistoryHeader, newHistoryHeader);

code = code.replace('{history.map((entry) => (', '{displayedHistory.map((entry) => (');

const historyFooter = `</button>
                ))}
              </div>`;
const newHistoryFooter = `</button>
                ))}
                {history.length > 3 && (
                  <button
                    onClick={() => setShowAllHistory(!showAllHistory)}
                    className="flex justify-center items-center gap-1 w-full py-1.5 mt-1 text-xs font-medium text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] transition-colors rounded hover:bg-[var(--color-surface-elevated)]"
                  >
                    {showAllHistory ? (
                      <><ChevronUp className="w-3 h-3" /> Show Less</>
                    ) : (
                      <><ChevronDown className="w-3 h-3" /> Show {history.length - 3} More</>
                    )}
                  </button>
                )}
              </div>`;

code = code.replace(historyFooter, newHistoryFooter);

fs.writeFileSync('frontend/src/components/InputPanel.jsx', code);
