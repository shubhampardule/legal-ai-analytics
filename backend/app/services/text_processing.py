from __future__ import annotations

import re
import unicodedata


SIGNATURE_NOT_VERIFIED_RE = re.compile(r"Signature Not Verified", re.IGNORECASE)
DIGITAL_SIGNATURE_RE = re.compile(
    r"Digitally signed by [A-Z][A-Z .]{1,80}", re.IGNORECASE
)
DATE_STAMP_RE = re.compile(r"Date:\s*[\d./:\- ]{6,40}", re.IGNORECASE)
EMAIL_LIKE_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.\w+\b", re.IGNORECASE)
HYPHENATED_LINEBREAK_RE = re.compile(r"([A-Za-z])-\s*\n\s*([A-Za-z])")
MULTI_NEWLINE_DASH_RE = re.compile(r"\s*--\s*")
NEWLINE_RE = re.compile(r"\s*\n\s*")
MULTISPACE_RE = re.compile(r"[ \t\f\v]+")
SENTENCE_RE = re.compile(r".+?(?:[.!?](?=\s+|$)|;(?=\s+)|$)", re.DOTALL)


def clean_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text or "")
    normalized = normalized.replace("\u00ad", "")
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = SIGNATURE_NOT_VERIFIED_RE.sub(" ", normalized)
    normalized = DIGITAL_SIGNATURE_RE.sub(" ", normalized)
    normalized = DATE_STAMP_RE.sub(" ", normalized)
    normalized = EMAIL_LIKE_RE.sub(" ", normalized)
    normalized = HYPHENATED_LINEBREAK_RE.sub(r"\1\2", normalized)
    normalized = MULTI_NEWLINE_DASH_RE.sub(" ", normalized)
    normalized = NEWLINE_RE.sub(" ", normalized)
    normalized = MULTISPACE_RE.sub(" ", normalized)
    return normalized.strip()


def extract_sentences(text: str) -> list[dict[str, object]]:
    sentences: list[dict[str, object]] = []
    for match in SENTENCE_RE.finditer(text or ""):
        raw_text = match.group(0)
        left_trim = len(raw_text) - len(raw_text.lstrip())
        right_trim = len(raw_text) - len(raw_text.rstrip())
        start_char = match.start() + left_trim
        end_char = match.end() - right_trim
        sentence_text = (text or "")[start_char:end_char].strip()
        if not sentence_text:
            continue
        if not any(char.isalnum() for char in sentence_text):
            continue
        sentences.append(
            {
                "text": sentence_text,
                "start_char": int(start_char),
                "end_char": int(end_char),
                "char_length": int(len(sentence_text)),
            }
        )

    if not sentences and (text or "").strip():
        stripped = (text or "").strip()
        sentences.append(
            {
                "text": stripped,
                "start_char": 0,
                "end_char": len(stripped),
                "char_length": len(stripped),
            }
        )
    return sentences

STATUTE_RE = re.compile(r'\b(section\s+\d+[a-z]?\s+of\s+(?:the\s+)?(?:[a-z]{3,20}\s+){1,4}act(?:\s+\d{4})?)\b', re.IGNORECASE)
ARTICLE_RE = re.compile(r'\b(article\s+\d+[a-z]?\s+of\s+(?:the\s+)?(?:constitution|companystitution)(?:\s+of\s+india)?)\b', re.IGNORECASE)
ACT_RE = re.compile(r'\b((?:the\s+)?(?:[a-z]{2,20}\s+){1,4}act(?:,?\s+\d{4})?)\b', re.IGNORECASE)
JUDGE_RE = re.compile(
    r"\b(?:hon'?ble\s+)?(?:mr\.?\s+|mrs\.?\s+|ms\.?\s+|dr\.?\s+)?(?:justice|judge|c\.?j\.?|j\.?|jj\.?)\s+[a-z][a-z.\-]*(?:\s+[a-z][a-z.\-]*){0,2}",
    re.IGNORECASE,
)

JUDGE_TITLE_WORDS = {
    "honble", "hon'ble", "mr", "mrs", "ms", "dr", "justice", "judge", "cj", "j", "jj",
}

JUDGE_STOPWORDS = {
    "in", "to", "of", "for", "and", "or", "the", "this", "that", "these", "those", "group",
    "would", "be", "done", "him", "her", "them", "union", "state", "petitioner", "respondent",
    "versus", "vs", "v",
}


def _clean_judge_candidate(raw: str) -> str | None:
    candidate = raw.strip(" ,;:.()[]{}\t\n\r")
    if not candidate:
        return None

    candidate = re.split(
        r"\s+(?=(?:hon'?ble|mr\.?|mrs\.?|ms\.?|dr\.?|justice|judge|c\.?j\.?|j\.?|jj\.?)(?:\s|$))",
        candidate,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]

    tokens = [tok for tok in candidate.replace("-", " ").split() if tok]
    if len(tokens) < 2:
        return None

    name_tokens: list[str] = []
    for tok in tokens:
        normalized = tok.strip(".,;:()[]{}\t\n\r")
        lowered = normalized.lower()
        if lowered in JUDGE_TITLE_WORDS:
            continue
        if lowered in JUDGE_STOPWORDS:
            return None
        if not re.fullmatch(r"[A-Za-z][A-Za-z.\-]*", normalized):
            return None
        if len(normalized) == 1 and normalized.lower() not in {"j", "c"}:
            return None
        name_tokens.append(normalized)

    if not (1 <= len(name_tokens) <= 3):
        return None

    if any(tok.lower() in JUDGE_STOPWORDS for tok in name_tokens):
        return None

    cleaned = " ".join(tokens).replace("C J", "C.J.")
    cleaned = re.sub(r"\s+", " ", cleaned).strip().title()

    # Guardrail: avoid sentence-like fragments pretending to be names.
    alpha_words = [w for w in re.findall(r"[A-Za-z]+", cleaned)]
    if len(alpha_words) > 4:
        return None

    return cleaned

def extract_legal_entities(text: str) -> dict[str, list[str]]:
    statutes = []

    for match in STATUTE_RE.finditer(text):
        res = match.group(1)
        res = re.sub(r'^(under|of|in|by|for|with|that|as)\s+', '', res, flags=re.IGNORECASE).strip().title()
        res = res.replace('Companystitution', 'Constitution')
        statutes.append(res)
    for match in ARTICLE_RE.finditer(text):
        res = match.group(1)
        res = re.sub(r'^(under|of|in|by|for|with|that|as)\s+', '', res, flags=re.IGNORECASE).strip().title()
        res = res.replace('Companystitution', 'Constitution')
        statutes.append(res)
    for match in ACT_RE.finditer(text):
        res = match.group(1)
        res = re.sub(r'^(under|of|in|by|for|with|that|as)\s+', '', res, flags=re.IGNORECASE).strip().title()
        if len(res.split()) > 10: continue
        statutes.append(res)

    judges = []
    for match in JUDGE_RE.finditer(text):
        cleaned = _clean_judge_candidate(match.group(0))
        if cleaned:
            judges.append(cleaned)

    def dedup(seq):
        seen = set()
        return [x for x in seq if not (x.lower() in seen or seen.add(x.lower()))]

    return {
        "statutes": dedup(statutes),
        "judges": dedup(judges)
    }
