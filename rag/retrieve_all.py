from __future__ import annotations

from langsmith import traceable

from rag.merge_results import merge_with_fallback
from rag.retrieve_primary import retrieve_primary
from rag.retrieve_secondary import retrieve_secondary
from rag.runtime_diagnostics import measure_stage
from rag.search_config import (
    ANSWER_MODE_NORM_ONLY,
    DEFAULT_ANSWER_MODE,
    DEFAULT_SOURCE_MODE,
    get_sources_for_mode,
    normalize_answer_mode,
    normalize_source_mode,
)


PRIMARY_SOURCE_IDS = {"590P", "611P"}
SECONDARY_SOURCE_IDS = {"590P_EXPL", "611P_EXPL"}


def split_sources_by_layer(source_ids: list[str]) -> tuple[list[str], list[str]]:
    """
    Split selected source identifiers into primary and secondary groups.
    """
    selected_sources = set(source_ids)

    primary_doc_ids = sorted(selected_sources & PRIMARY_SOURCE_IDS)
    secondary_doc_ids = sorted(selected_sources & SECONDARY_SOURCE_IDS)

    return primary_doc_ids, secondary_doc_ids


def get_effective_source_ids(
    source_mode: str,
    answer_mode: str,
) -> list[str]:
    """
    Resolve source_mode and answer_mode into concrete source identifiers.

    For answer_mode='norm_only', we restrict search to primary normative sources.
    """
    source_mode = normalize_source_mode(source_mode)
    answer_mode = normalize_answer_mode(answer_mode)

    source_ids = get_sources_for_mode(source_mode)

    if answer_mode == ANSWER_MODE_NORM_ONLY:
        source_ids = [
            source_id for source_id in source_ids
            if source_id in PRIMARY_SOURCE_IDS
        ]

    return source_ids


def summarize_retrieval_output(result: dict) -> dict:
    """
    Prepare a compact retrieval summary for LangSmith.

    Do not log full document texts or embeddings.
    The original RAG result remains unchanged.
    """
    return {
        "source_mode": result.get("source_mode"),
        "answer_mode": result.get("answer_mode"),
        "source_ids": result.get("source_ids", []),
        "primary_doc_ids": result.get("primary_doc_ids", []),
        "secondary_doc_ids": result.get("secondary_doc_ids", []),
        "result_counts": {
            key: len(value)
            for key, value in result.items()
            if isinstance(value, (list, tuple))
        },
    }
@traceable(
    name="retrieve_all",
    run_type="chain",
    tags=["rag", "retrieval"],
    process_outputs=summarize_retrieval_output,
)
def retrieve_all(
    query: str,
    source_mode: str = DEFAULT_SOURCE_MODE,
    answer_mode: str = DEFAULT_ANSWER_MODE,
    top_k_primary: int = 5,
    top_k_secondary: int = 5,
    primary_k_semantic: int = 15,
    primary_k_keyword: int = 15,
    primary_final_k: int = 30,
    secondary_k_semantic: int = 15,
    secondary_k_keyword: int = 15,
    secondary_final_k: int = 30,
) -> dict:
    source_ids = get_effective_source_ids(
        source_mode=source_mode,
        answer_mode=answer_mode,
    )

    primary_doc_ids, secondary_doc_ids = split_sources_by_layer(source_ids)


    primary_results = []
    secondary_results = []

    if primary_doc_ids:
        with measure_stage("retrieval.primary"):
            primary_results = retrieve_primary(
                query=query,
                k=top_k_primary,
                k_semantic=primary_k_semantic,
                k_keyword=primary_k_keyword,
                final_k=primary_final_k,
                doc_ids=primary_doc_ids,
            )

    if secondary_doc_ids:
        with measure_stage("retrieval.secondary"):
            secondary_results = retrieve_secondary(
                query=query,
                k=top_k_secondary,
                k_semantic=secondary_k_semantic,
                k_keyword=secondary_k_keyword,
                final_k=secondary_final_k,
                doc_ids=secondary_doc_ids,
            )

    with measure_stage("retrieval.merge"):
        merged_result = merge_with_fallback(
            primary_results=primary_results,
            secondary_results=secondary_results,
            top_k_primary=top_k_primary,
            top_k_secondary=top_k_secondary,
        )

    merged_result["source_mode"] = source_mode
    merged_result["answer_mode"] = answer_mode
    merged_result["source_ids"] = source_ids
    merged_result["primary_doc_ids"] = primary_doc_ids
    merged_result["secondary_doc_ids"] = secondary_doc_ids

    return merged_result