from __future__ import annotations

from pathlib import Path
from typing import Iterable

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from rag.runtime_diagnostics import measure_stage

BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = BASE_DIR / "data" / "chroma"
COLLECTION_NAME = "cbr_llp_copilot"
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"

_EMBEDDINGS: HuggingFaceEmbeddings | None = None
_VECTORSTORE: Chroma | None = None



def get_embeddings() -> HuggingFaceEmbeddings:
    global _EMBEDDINGS

    if _EMBEDDINGS is None:
        with measure_stage("embeddings.initialization"):
            _EMBEDDINGS = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL
            )

    return _EMBEDDINGS



def get_vectorstore() -> Chroma:
    global _VECTORSTORE

    if _VECTORSTORE is None:
        embeddings = get_embeddings()

        with measure_stage("chroma.initialization"):
            _VECTORSTORE = Chroma(
                collection_name=COLLECTION_NAME,
                embedding_function=embeddings,
                persist_directory=str(CHROMA_DIR),
            )

    return _VECTORSTORE

def normalize_text(text: str) -> str:
    return (text or "").strip().lower()


def tokenize_query(query: str) -> list[str]:
    text = normalize_text(query)

    for ch in [",", ".", ":", ";", "(", ")", "/", "\\", "\n", "\t", "-", "—", "–", "?", "!"]:
        text = text.replace(ch, " ")

    return [t for t in text.split() if len(t) >= 3]


def get_doc_identity(doc: Document) -> tuple:
    md = doc.metadata or {}
    return (
        md.get("doc_id"),
        md.get("page_num"),
        md.get("chunk_num"),
        md.get("row_id"),
        md.get("block_id"),
        (doc.page_content or "")[:200],
    )


def deduplicate_documents(docs: Iterable[Document]) -> list[Document]:
    seen = set()
    result = []

    for doc in docs:
        key = get_doc_identity(doc)
        if key in seen:
            continue
        seen.add(key)
        result.append(doc)

    return result


def filter_documents_by_doc_ids(
    docs: list[Document],
    doc_ids: list[str] | None = None,
) -> list[Document]:
    """
    Filter documents by metadata['doc_id'].

    If doc_ids is None or empty, return documents unchanged.
    """
    if not doc_ids:
        return docs

    allowed_doc_ids = set(doc_ids)

    return [
        doc for doc in docs
        if (doc.metadata or {}).get("doc_id") in allowed_doc_ids
    ]



def semantic_search(
    query: str,
    doc_ids: list[str] | None = None,
    k: int = 10,
    fetch_k: int = 30,
) -> list[Document]:
    db = get_vectorstore()

    with measure_stage("retrieval.semantic_mmr"):
        candidates = db.max_marginal_relevance_search(
            "query: " + query,
            k=fetch_k,
            fetch_k=max(fetch_k, k),
        )

    with measure_stage("retrieval.semantic_filtering"):
        candidates = filter_documents_by_doc_ids(
            docs=candidates,
            doc_ids=doc_ids,
        )

    return candidates[:k]


def score_doc_by_query(doc: Document, query: str) -> int:
    tokens = tokenize_query(query)

    text = normalize_text(doc.page_content)
    md = doc.metadata or {}

    metadata_text_parts = [
        str(md.get("doc_title", "")),
        str(md.get("theme", "")),
        str(md.get("chapter", "")),
        str(md.get("clauses", "")),
        str(md.get("keywords", "")),
        str(md.get("topic", "")),
        str(md.get("answer_date", "")),
    ]
    metadata_text = normalize_text(" ".join(metadata_text_parts))

    score = 0
    for token in tokens:
        if token in text:
            score += 2
        if token in metadata_text:
            score += 3

    return score


def keyword_search(
    query: str,
    docs: list[Document],
    doc_ids: list[str] | None = None,
    k: int = 10,
    min_score: int = 1,
) -> list[Document]:
    filtered_docs = filter_documents_by_doc_ids(
        docs=docs,
        doc_ids=doc_ids,
    )

    scored: list[tuple[int, Document]] = []

    for doc in filtered_docs:
        score = score_doc_by_query(doc, query)
        if score >= min_score:
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

    return [doc for score, doc in scored[:k]]



def hybrid_search(
    query: str,
    docs_for_keyword_search: list[Document],
    doc_ids_for_semantic: list[str] | None = None,
    doc_ids_for_keyword: list[str] | None = None,
    k_semantic: int = 10,
    k_keyword: int = 10,
    final_k: int = 15,
) -> list[Document]:
    semantic_docs = semantic_search(
        query=query,
        doc_ids=doc_ids_for_semantic,
        k=k_semantic,
        fetch_k=max(20, k_semantic * 3),
    )

    with measure_stage("retrieval.keyword_search"):
        keyword_docs = keyword_search(
            query=query,
            docs=docs_for_keyword_search,
            doc_ids=doc_ids_for_keyword,
            k=k_keyword,
            min_score=1,
        )

    with measure_stage("retrieval.deduplication"):
        merged = deduplicate_documents(
            keyword_docs + semantic_docs
        )

    return merged[:final_k]