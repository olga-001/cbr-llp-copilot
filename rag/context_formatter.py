from langsmith import traceable

from typing import Any, Dict, List

from rag.metadata_utils import normalize_retrieval_result_item


def clean_page_content(text: str) -> str:
    """
    Минимальная очистка текста перед передачей в LLM.
    Ничего агрессивно не режем, только убираем лишние пробелы по краям.
    """
    if not text:
        return ""
    return text.strip()


def build_context_block(item: Dict[str, Any], index: int) -> str:
    """
    Преобразует один retrieval item в единый текстовый блок для prompt.

    Ожидает item формата:
    {
        "document": Document(...),
        "rank_score": ...,
        "source_doc": ...,
        "source_type": ...
    }

    Возвращает блок вида:

    [DOC_1]
    source: 590-P
    title: Положение Банка России №590-П
    source_type: primary
    locator: п. 3.12
    priority: 1
    rank_score: 13
    text:
    ...
    """
    normalized_item = normalize_retrieval_result_item(item)

    doc = normalized_item["document"]
    metadata = doc.metadata or {}

    source = metadata.get("source", "unknown")
    title = metadata.get("title", source)
    source_type = metadata.get("source_type", "unknown")
    locator = metadata.get("locator", "n/a")
    priority = metadata.get("priority", "n/a")
    rank_score = normalized_item.get("rank_score")

    text = clean_page_content(getattr(doc, "page_content", ""))

    block = (
        f"[DOC_{index}]\n"
        f"source: {source}\n"
        f"title: {title}\n"
        f"source_type: {source_type}\n"
        f"locator: {locator}\n"
        f"priority: {priority}\n"
        f"rank_score: {rank_score}\n"
        f"text:\n{text}\n"
    )
    return block


def build_context_blocks(items: List[Dict[str, Any]]) -> str:
    """
    Собирает все selected_items в один текстовый context для LLM.
    """
    blocks = []
    for idx, item in enumerate(items, start=1):
        blocks.append(build_context_block(item, idx))
    return "\n\n".join(blocks)


def build_citation_candidates(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Готовит кандидатов для citations из selected_items.
    Это еще не финальный citations output для пользователя, а
    удобная структурированная основа для answer-layer.
    """
    citation_candidates = []

    for idx, item in enumerate(items, start=1):
        normalized_item = normalize_retrieval_result_item(item)
        doc = normalized_item["document"]
        metadata = doc.metadata or {}
        text = clean_page_content(getattr(doc, "page_content", ""))

        citation_candidates.append(
            {
                "doc_ref": f"DOC_{idx}",
                "source": metadata.get("source", "unknown"),
                "title": metadata.get("title", "unknown"),
                "locator": metadata.get("locator", "n/a"),
                "source_type": metadata.get("source_type", "unknown"),
                "priority": metadata.get("priority"),
                "rank_score": normalized_item.get("rank_score"),
                "snippet": text[:500],
                "full_text": text,
            }
        )

    return citation_candidates


# ---------------------------------------------------------
# LangSmith: compact Context Formatting tracing
# ---------------------------------------------------------

def summarize_context_input(inputs: dict) -> dict:
    """
    Record the number of selected items without document texts.
    """
    items = inputs.get("items") or []

    return {
        "selected_items_count": len(items),
    }


def summarize_context_output(result: dict) -> dict:
    """
    Record context size and citation statistics.
    The original result is not modified.
    """
    return {
        "normalized_items_count": len(
            result.get("normalized_items") or []
        ),
        "context_length_chars": len(
            result.get("context_blocks") or ""
        ),
        "citation_candidates_count": len(
            result.get("citation_candidates") or []
        ),
        "documents_count": len(
            result.get("documents") or []
        ),
    }

@traceable(
    name="prepare_context_package",
    run_type="chain",
    tags=["rag", "context"],
    process_inputs=summarize_context_input,
    process_outputs=summarize_context_output,
)
def prepare_context_package(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Главная функция модуля.

    Принимает selected_items после selection layer и возвращает пакет:
    {
        "normalized_items": [...],
        "context_blocks": "...",
        "citation_candidates": [...],
        "documents": [...]
    }

    Это будет прямой вход для answer_pipeline.py.
    """
    normalized_items = [normalize_retrieval_result_item(item) for item in items]
    context_blocks = build_context_blocks(normalized_items)
    citation_candidates = build_citation_candidates(normalized_items)
    documents = [item["document"] for item in normalized_items if item.get("document") is not None]

    return {
        "normalized_items": normalized_items,
        "context_blocks": context_blocks,
        "citation_candidates": citation_candidates,
        "documents": documents,
    }