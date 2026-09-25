import re

STOP_WORDS = {
    "the",
    "this",
    "that",
    "with",
    "from",
    "have",
    "error",
    "what",
    "where",
    "how",
    "when",
    "which",
    "who",
    "whom",
    "will",
    "would",
    "shall",
    "should",
    "can",
    "could",
    "about",
    "into",
    "over",
    "after",
    "http",
    "https",
    "get",
    "post",
}


def extract_ascii_identifiers(
    query: str, min_len: int = 3, stop_words: set[str] | None = None
) -> list[str]:
    """
    Extracts technical identifiers from a query string.
    """
    if stop_words is None:
        stop_words = STOP_WORDS

    raw_tokens = re.findall(r"[a-zA-Z0-9_\-\.\/]+", query)

    identifiers = []
    seen = set()

    for tok in raw_tokens:
        tok_lower = tok.lower().strip("_-/,")
        # Strip trailing dot (sentence ending period) while preserving internal dots like in file extensions
        tok_lower = tok_lower.rstrip(".")
        if not tok_lower:
            continue

        if tok_lower in seen:
            continue

        if tok_lower.isdigit():
            continue

        if tok_lower in stop_words:
            continue

        has_alpha = any(c.isalpha() for c in tok_lower)
        has_digit = any(c.isdigit() for c in tok_lower)
        has_symbol = any(c in "_-./" for c in tok_lower)

        is_alphanumeric_mixed = has_alpha and has_digit
        is_pure_alpha = has_alpha and not has_digit and not has_symbol

        keep = False
        if is_alphanumeric_mixed:
            keep = True
        elif has_symbol:
            keep = True
        elif is_pure_alpha and len(tok_lower) >= 4:
            keep = True

        # check min_len only if it's pure alpha without symbols, or something.
        # Requirements:
        # - Extract raw candidate tokens matching `[a-zA-Z0-9_\-\.\/]+` (length >= min_len or short alphanumeric tokens).
        if len(tok_lower) < min_len and not is_alphanumeric_mixed and not has_symbol:
            keep = False

        if keep:
            identifiers.append(tok_lower)
            seen.add(tok_lower)

    return identifiers


def calculate_containment(token: str, doc_lower: str, doc_3grams: set[str]) -> float:
    """
    Calculates the 3-gram containment ratio of a token in a document.
    """
    if len(token) < 3:
        return 1.0 if token in doc_lower else 0.0

    token_3grams = {token[i : i + 3] for i in range(len(token) - 2)}
    if not token_3grams:
        return 0.0

    intersection = token_3grams.intersection(doc_3grams)
    return float(len(intersection)) / float(len(token_3grams))


class AsciiMatcher:
    def __init__(self, min_token_len: int = 3, stop_words: set[str] | None = None):
        self.min_token_len = min_token_len
        self.stop_words = stop_words if stop_words is not None else STOP_WORDS

    def extract_identifiers(self, query: str) -> list[str]:
        return extract_ascii_identifiers(query, self.min_token_len, self.stop_words)

    def score_document(self, query_tokens: list[str], document: str) -> float:
        """Return average containment ratio (0.0 to 1.0) of query tokens in document."""
        if not query_tokens:
            return 0.0

        doc_lower = document.lower()
        doc_3grams = {doc_lower[i : i + 3] for i in range(len(doc_lower) - 2)}

        total_score = 0.0
        for token in query_tokens:
            total_score += calculate_containment(token, doc_lower, doc_3grams)

        return total_score / len(query_tokens)

    def score_documents(self, query: str, documents: list[str]) -> list[float]:
        """Extract query tokens once, and calculate containment scores for all documents."""
        query_tokens = self.extract_identifiers(query)
        if not query_tokens:
            return [0.0] * len(documents)

        return [self.score_document(query_tokens, doc) for doc in documents]
