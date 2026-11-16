from typing import Any, Dict, List, Optional

from langsmith import traceable

def deduplicate_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Убирает дубли retrieval items по устойчивому ключу.

    Приоритет ключей дедупликации:
    1. chunk_id
    2. doc_id + locator + первые символы текста
    3. source_doc + rank_score + первые символы текста

    Это нужно, чтобы один и тот же фрагмент не попадал в context несколько раз.
    """
    seen = set()
    unique_items = []

    for item in items:
        doc = item.get("document")
        metadata = getattr(doc, "metadata", {}) or {}
        page_content = (getattr(doc, "page_content", "") or "").strip()

        chunk_id = metadata.get("chunk_id")
        doc_id = metadata.get("doc_id")
        locator = metadata.get("locator")
        source_doc = item.get("source_doc")
        rank_score = item.get("rank_score")

        if chunk_id:
            key = ("chunk_id", chunk_id)
        elif doc_id or locator:
            key = ("doc_locator", doc_id, locator, page_content[:200])
        else:
            key = ("fallback", source_doc, rank_score, page_content[:200])

        if key in seen:
            continue

        seen.add(key)
        unique_items.append(item)

    return unique_items


def sort_items_by_rank(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Сортирует retrieval items по убыванию rank_score.
    Если score отсутствует, такой item уходит ниже.
    """
    return sorted(
        items,
        key=lambda x: x.get("rank_score") if x.get("rank_score") is not None else -10**9,
        reverse=True,
    )


def filter_items_by_source_type(
    items: List[Dict[str, Any]],
    source_type: str,
) -> List[Dict[str, Any]]:
    """
    Оставляет только items нужного source_type.
    """
    result = []

    for item in items:
        item_source_type = item.get("source_type")

        if item_source_type is None:
            doc = item.get("document")
            metadata = getattr(doc, "metadata", {}) or {}
            item_source_type = metadata.get("source_type")

        if item_source_type == source_type:
            result.append(item)

    return result


def select_top_items(
    items: List[Dict[str, Any]],
    k: int,
) -> List[Dict[str, Any]]:
    """
    Берет top-k после сортировки и дедупликации.
    """
    items = sort_items_by_rank(items)
    items = deduplicate_items(items)
    return items[:k]


# ---------------------------------------------------------
# LangSmith: compact Selection tracing
# ---------------------------------------------------------

def summarize_selection_input(inputs: dict) -> dict:
    """
    Record Selection parameters and retrieval counts
    without logging full document texts.
    """
    retrieval = inputs.get("retrieval_result") or {}

    return {
        "retrieval_counts": {
            "primary": len(retrieval.get("primary_results") or []),
            "secondary": len(retrieval.get("secondary_results") or []),
            "combined": len(retrieval.get("combined_results") or []),
        },
        "primary_k": inputs.get("primary_k", 3),
        "secondary_k": inputs.get("secondary_k", 2),
        "tertiary_k": inputs.get("tertiary_k", 0),
        "use_combined_fallback": inputs.get(
            "use_combined_fallback", True
        ),
    }


def summarize_selection_output(result: dict) -> dict:
    """
    Record the Selection outcome without document contents.
    The original result returned to the RAG Pipeline is unchanged.
    """
    return {
        "selection_mode": result.get("selection_mode"),
        "selected_counts": {
            "total": len(result.get("selected_items") or []),
            "primary": len(result.get("selected_primary") or []),
            "secondary": len(result.get("selected_secondary") or []),
            "tertiary": len(result.get("selected_tertiary") or []),
        },
        "has_primary": result.get("has_primary"),
        "has_secondary": result.get("has_secondary"),
    }
@traceable(
    name="select_context_items",
    run_type="chain",
    tags=["rag", "selection"],
    process_inputs=summarize_selection_input,
    process_outputs=summarize_selection_output,
)
def select_context_items(
    retrieval_result: Dict[str, Any],
    primary_k: int = 3,
    secondary_k: int = 2,
    tertiary_k: int = 0,
    use_combined_fallback: bool = True,
) -> Dict[str, Any]:
    """
    Основная функция отбора контекста для answer-layer.

    Вход:
        retrieval_result = результат retrieve_all(question), где ожидаются ключи:
        - primary_results
        - secondary_results
        - combined_results
        - mode (optional)

    Логика:
    1. Берем top primary_results
    2. Берем top secondary_results
    3. При необходимости берем tertiary из combined_results
    4. Если primary пуст, а use_combined_fallback=True, добираем из combined_results
    5. Возвращаем:
       - selected_items
       - selected_primary
       - selected_secondary
       - selected_tertiary
       - has_primary
       - has_secondary
       - selection_mode
    """
    primary_results = retrieval_result.get("primary_results", []) or []
    secondary_results = retrieval_result.get("secondary_results", []) or []
    combined_results = retrieval_result.get("combined_results", []) or []

    # На случай, если tertiary отдельно пока не выделяются
    tertiary_results = filter_items_by_source_type(combined_results, "tertiary")

    selected_primary = select_top_items(primary_results, primary_k)
    selected_secondary = select_top_items(secondary_results, secondary_k)
    selected_tertiary = select_top_items(tertiary_results, tertiary_k) if tertiary_k > 0 else []

    selected_items = selected_primary + selected_secondary + selected_tertiary
    selected_items = deduplicate_items(selected_items)

    selection_mode = "primary_plus_secondary"

    if not selected_primary and selected_secondary:
        selection_mode = "secondary_only"

        if use_combined_fallback:
            fallback_items = select_top_items(combined_results, primary_k + secondary_k + tertiary_k)
            selected_items = deduplicate_items(fallback_items)
            selection_mode = "secondary_only_with_combined_fallback"

    elif not selected_primary and not selected_secondary:
        if use_combined_fallback:
            fallback_items = select_top_items(combined_results, primary_k + secondary_k + tertiary_k)
            selected_items = deduplicate_items(fallback_items)
            selection_mode = "combined_fallback_only"
        else:
            selected_items = []
            selection_mode = "empty"

    return {
        "selected_items": selected_items,
        "selected_primary": selected_primary,
        "selected_secondary": selected_secondary,
        "selected_tertiary": selected_tertiary,
        "has_primary": len(selected_primary) > 0,
        "has_secondary": len(selected_secondary) > 0,
        "selection_mode": selection_mode,
    }


def items_to_documents(items: List[Dict[str, Any]]) -> List[Any]:
    """
    Из retrieval items достает только Document-объекты.
    """
    docs = []
    for item in items:
        doc = item.get("document")
        if doc is not None:
            docs.append(doc)
    return docs


def get_best_primary_item(retrieval_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Возвращает лучший primary item или None.
    """
    primary_results = retrieval_result.get("primary_results", []) or []
    selected = select_top_items(primary_results, 1)
    return selected[0] if selected else None


def get_best_secondary_item(retrieval_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Возвращает лучший secondary item или None.
    """
    secondary_results = retrieval_result.get("secondary_results", []) or []
    selected = select_top_items(secondary_results, 1)
    return selected[0] if selected else None