from __future__ import annotations

from typing import Dict, List

from langchain_core.documents import Document

from rag.reranker import rank_documents
from rag.retrieve_hybrid import hybrid_search, get_vectorstore


SECONDARY_DOC_IDS = ["590P_EXPL", "611P_EXPL"]


def resolve_secondary_doc_ids(doc_ids: list[str] | None = None) -> list[str]:
    """
    Return allowed secondary doc_ids.

    If doc_ids is provided, keep only secondary source ids.
    If doc_ids is None or empty, use all secondary sources.
    """
    if not doc_ids:
        return SECONDARY_DOC_IDS

    allowed = set(SECONDARY_DOC_IDS)

    return [
        doc_id for doc_id in doc_ids
        if doc_id in allowed
    ]


def get_all_secondary_docs(doc_ids: list[str] | None = None) -> List[Document]:
    selected_doc_ids = resolve_secondary_doc_ids(doc_ids)

    if not selected_doc_ids:
        return []

    db = get_vectorstore()
    all_docs = db.get(include=["documents", "metadatas"])

    documents: List[Document] = []
    for content, metadata in zip(all_docs["documents"], all_docs["metadatas"]):
        if (metadata or {}).get("doc_id") in selected_doc_ids:
            documents.append(
                Document(
                    page_content=content,
                    metadata=metadata,
                )
            )

    return documents


def retrieve_secondary(
    query: str,
    k: int = 5,
    k_semantic: int = 15,
    k_keyword: int = 15,
    final_k: int = 30,
    doc_ids: list[str] | None = None,
) -> List[Dict]:
    selected_doc_ids = resolve_secondary_doc_ids(doc_ids)

    if not selected_doc_ids:
        return []

    all_secondary_docs = get_all_secondary_docs(doc_ids=selected_doc_ids)

    candidates = hybrid_search(
        query=query,
        docs_for_keyword_search=all_secondary_docs,
        doc_ids_for_semantic=selected_doc_ids,
        doc_ids_for_keyword=selected_doc_ids,
        k_semantic=k_semantic,
        k_keyword=k_keyword,
        final_k=final_k,
    )

    if not candidates:
        return []

    ranked = rank_documents(candidates, query)

    results: List[Dict] = []
    for score, doc in ranked[:k]:
        results.append({
            "source_doc": (doc.metadata or {}).get("doc_id"),
            "document": doc,
            "rank_score": score,
        })

    return results