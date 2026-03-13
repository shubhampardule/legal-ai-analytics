const API_PREFIX = "/api/v1";
const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "").replace(/\/$/, "");

function buildUrl(path) {
  return API_BASE_URL ? `${API_BASE_URL}${path}` : path;
}

async function parseEnvelope(response) {
  const body = await response.json();
  if (!response.ok || body.status !== "ok") {
    const message =
      body?.error?.message || `Request failed with status ${response.status}`;
    throw new Error(message);
  }
  return body;
}

export async function fetchHealth() {
  const response = await fetch(buildUrl("/health"));
  return parseEnvelope(response);
}

export async function fetchMeta() {
  const response = await fetch(buildUrl(`${API_PREFIX}/meta`));
  return parseEnvelope(response);
}

export async function analyzeCase(payload) {
  const response = await fetch(buildUrl(`${API_PREFIX}/analyze`), {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
  return parseEnvelope(response);
}

export async function chatWithCase(payload) {
  const response = await fetch(buildUrl(`${API_PREFIX}/chat-case`), {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
  return parseEnvelope(response);
}

export async function getCaseDetails(caseId) {
  const response = await fetch(buildUrl(`${API_PREFIX}/cases/${caseId}`));
  return parseEnvelope(response);
}

export async function listCases(limit = 50, offset = 0, query = "", filters = {}) {
  const url = new URL(buildUrl(`${API_PREFIX}/cases`), window.location.origin);
  url.searchParams.append("limit", limit);
  url.searchParams.append("offset", offset);
  if (query) {
    url.searchParams.append("query", query);
  }
  if (filters?.outcome) {
    url.searchParams.append("outcome", filters.outcome);
  }
  if (Number.isInteger(filters?.year_from)) {
    url.searchParams.append("year_from", String(filters.year_from));
  }
  if (Number.isInteger(filters?.year_to)) {
    url.searchParams.append("year_to", String(filters.year_to));
  }
  const response = await fetch(url.toString());
  return parseEnvelope(response);
}
