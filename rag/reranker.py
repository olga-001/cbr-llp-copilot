from __future__ import annotations

import re
from collections import Counter
from langchain_core.documents import Document


STOPWORDS = {
    "и", "или", "в", "во", "на", "по", "с", "со", "к", "ко", "от", "до",
    "для", "о", "об", "не", "но", "а", "как", "если", "ли", "то", "же",
    "что", "это", "при", "из", "у", "над", "под", "без", "за", "так",
    "его", "ее", "их", "бы", "быть", "также", "либо", "который", "которая",
    "которые", "данный", "данная", "данные", "этот", "эта", "эти"
}


def normalize_text(text: str) -> str:
    return (text or "").strip().lower()


def tokenize(text: str) -> list[str]:
    text = normalize_text(text)
    text = re.sub(r"[^\w\s\.\-]", " ", text, flags=re.UNICODE)
    return [t for t in text.split() if len(t) >= 3 and t not in STOPWORDS]


def extract_number_patterns(text: str) -> list[str]:
    """
    Ищем номера пунктов / подпунктов:
    3.9, 3.9.2, 6.2.2 и т.п.
    Даты формата 15.02.2022 исключаем.
    """
    candidates = re.findall(r"\b\d+\.\d+(?:\.\d+)?\b", text or "")
    result = []

    for c in candidates:
        parts = c.split(".")
        if len(parts) == 3 and len(parts[2]) == 4:
            continue
        result.append(c)

    return result


def build_metadata_text(doc: Document) -> str:
    md = doc.metadata or {}

    parts = [
        str(md.get("doc_id", "")),
        str(md.get("doc_title", "")),
        str(md.get("theme", "")),
        str(md.get("chapter", "")),
        str(md.get("clauses", "")),
        str(md.get("keywords", "")),
        str(md.get("topic", "")),
        str(md.get("answer_date", "")),
        str(md.get("source_file", "")),
    ]
    return normalize_text(" ".join(parts))


def extract_phrases(text: str, min_words: int = 2, max_words: int = 3) -> list[str]:
    tokens = tokenize(text)
    phrases = []

    for n in range(min_words, max_words + 1):
        for i in range(len(tokens) - n + 1):
            phrase = " ".join(tokens[i:i + n])
            phrases.append(phrase)

    return phrases


def token_overlap_score(query: str, doc: Document) -> int:
    query_tokens = tokenize(query)
    if not query_tokens:
        return 0

    doc_text = normalize_text(doc.page_content)
    md_text = build_metadata_text(doc)

    score = 0
    query_counts = Counter(query_tokens)

    for token, count in query_counts.items():
        token_score = 0

        if token in doc_text:
            token_score += 2
        if token in md_text:
            token_score += 3

        if len(token) >= 8:
            token_score += 1

        score += token_score * count

    return score


def phrase_overlap_score(query: str, doc: Document) -> int:
    phrases = extract_phrases(query, min_words=2, max_words=3)
    if not phrases:
        return 0

    doc_text = normalize_text(doc.page_content)
    md_text = build_metadata_text(doc)

    score = 0
    for phrase in phrases:
        if phrase in doc_text:
            score += 5
        if phrase in md_text:
            score += 6

    return score


def clause_match_score(query: str, doc: Document) -> int:
    query_clauses = set(extract_number_patterns(query))
    if not query_clauses:
        return 0

    doc_text = normalize_text(doc.page_content)
    md_text = build_metadata_text(doc)

    score = 0
    for clause in query_clauses:
        if clause in doc_text:
            score += 8
        if clause in md_text:
            score += 10

    return score


def regulation_match_score(query: str, doc: Document) -> int:
    query_lower = normalize_text(query)
    md_text = build_metadata_text(doc)
    doc_text = normalize_text(doc.page_content)

    score = 0

    if "590" in query_lower and ("590" in md_text or "590" in doc_text):
        score += 5

    if "611" in query_lower and ("611" in md_text or "611" in doc_text):
        score += 5

    return score


def document_specificity_bonus(doc: Document) -> int:
    md = doc.metadata or {}
    score = 0

    if md.get("clauses"):
        score += 2
    if md.get("theme"):
        score += 1
    if md.get("keywords"):
        score += 2
    if md.get("chapter"):
        score += 1
    if md.get("topic"):
        score += 1

    return score


def low_overlap_penalty(query: str, doc: Document) -> int:
    """
    Очень мягкий штраф только за слабое совпадение с query.
    Без списков фиксированных фраз.
    """
    query_tokens = set(tokenize(query))
    if not query_tokens:
        return 0

    doc_text = normalize_text(doc.page_content)
    md_text = build_metadata_text(doc)

    overlap = 0
    for token in query_tokens:
        if token in doc_text or token in md_text:
            overlap += 1

    if overlap == 0:
        return 3
    if overlap == 1:
        return 1
    return 0


def score_document(doc: Document, query: str) -> int:
    score = 0
    score += token_overlap_score(query, doc)
    score += phrase_overlap_score(query, doc)
    score += clause_match_score(query, doc)
    score += regulation_match_score(query, doc)
    score += document_specificity_bonus(doc)
    score -= low_overlap_penalty(query, doc)
    return score


def rank_documents(docs: list[Document], query: str) -> list[tuple[int, Document]]:
    scored = []
    for doc in docs:
        score = score_document(doc, query)
        scored.append((score, doc))

    scored.sort(
        key=lambda x: (
            -x[0],
            (x[1].metadata or {}).get("page_num", 9999),
            (x[1].metadata or {}).get("chunk_num", 9999),
            (x[1].metadata or {}).get("row_id", 999999),
            (x[1].metadata or {}).get("block_id", 999999),
        )
    )
    return scored


def rank_items(items: list[dict], query: str) -> list[dict]:
    scored_items = []

    for item in items:
        doc = item["document"]
        score = score_document(doc, query)

        new_item = item.copy()
        new_item["rank_score"] = score
        scored_items.append(new_item)

    scored_items.sort(
        key=lambda x: (
            -x["rank_score"],
            (x["document"].metadata or {}).get("page_num", 9999),
            (x["document"].metadata or {}).get("chunk_num", 9999),
            (x["document"].metadata or {}).get("row_id", 999999),
            (x["document"].metadata or {}).get("block_id", 999999),
        )
    )

    return scored_items