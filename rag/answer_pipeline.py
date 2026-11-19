
from typing import Any, Dict

from langsmith import traceable

from rag.answer_contract import AnswerResult
from rag.answer_generator import generate_final_answer
from rag.context_formatter import prepare_context_package
from rag.retrieve_all import retrieve_all
from rag.runtime_diagnostics import measure_stage
from rag.search_config import (
    ANSWER_MODES,
    DEFAULT_ANSWER_MODE,
    DEFAULT_SOURCE_MODE,
    SOURCE_MODES,
)
from rag.selection_utils import select_context_items


def normalize_source_mode(source_mode: str) -> str:
    """
    Validate source_mode and return a safe internal source mode key.

    If an unknown source_mode is passed, fallback to DEFAULT_SOURCE_MODE.
    """
    if source_mode in SOURCE_MODES:
        return source_mode

    return DEFAULT_SOURCE_MODE


def normalize_answer_mode(answer_mode: str) -> str:
    """
    Validate answer_mode and return a safe internal answer mode key.

    If an unknown answer_mode is passed, fallback to DEFAULT_ANSWER_MODE.
    """
    if answer_mode in ANSWER_MODES:
        return answer_mode

    return DEFAULT_ANSWER_MODE


# ---------------------------------------------------------
# LangSmith: compact Pipeline tracing
# ---------------------------------------------------------

def summarize_pipeline_output(result: Dict[str, Any]) -> dict:
    """
    Record compact pipeline statistics in LangSmith.
    Do not duplicate document texts or full context.
    """
    retrieval = result.get("retrieval_result") or {}
    selection = result.get("selection") or {}
    context = result.get("context_package") or {}
    answer = result.get("answer_result")

    return {
        "source_mode": result.get("source_mode"),
        "answer_mode": result.get("answer_mode"),
        "retrieval_counts": {
            "primary": len(retrieval.get("primary_results") or []),
            "secondary": len(retrieval.get("secondary_results") or []),
            "combined": len(retrieval.get("combined_results") or []),
        },
        "selection_mode": selection.get("selection_mode"),
        "selected_items_count": len(
            selection.get("selected_items") or []
        ),
        "context_length_chars": len(
            context.get("context_blocks") or ""
        ),
        "citation_candidates_count": len(
            context.get("citation_candidates") or []
        ),
        "answer_length_chars": len(
            answer.answer or ""
        ) if answer is not None else 0,
        "citations_count": len(
            answer.citations or []
        ) if answer is not None else 0,
        "grounded": (
            answer.grounded if answer is not None else None
        ),
    }

@traceable(
    name="run_answer_pipeline",
    run_type="chain",
    tags=["rag", "pipeline"],
    process_outputs=summarize_pipeline_output,
)
def run_answer_pipeline(
    question: str,
    source_mode: str = DEFAULT_SOURCE_MODE,
    answer_mode: str = DEFAULT_ANSWER_MODE,
    primary_k: int = 3,
    secondary_k: int = 2,
    tertiary_k: int = 0,
    use_combined_fallback: bool = True,
    max_citations: int = 5,
) -> Dict[str, Any]:
    """
    Основной orchestration pipeline.

    Шаги:
    1. Retrieval
    2. Selection
    3. Context formatting
    4. LLM answer generation
    5. Возврат полного пакета для тестов / UI / отладки

    На текущем шаге Этапа 9 pipeline уже принимает source_mode и answer_mode,
    валидирует их и возвращает в результате.

    Важно:
    source_mode и answer_mode пока НЕ передаются в retrieve_all(),
    потому что retrieval-layer будет расширен следующим шагом.

    Формат результата:
    {
        "question": ...,
        "source_mode": ...,
        "answer_mode": ...,
        "retrieval_result": ...,
        "selection": ...,
        "context_package": ...,
        "answer_result": AnswerResult
    }
    """
    source_mode = normalize_source_mode(source_mode)
    answer_mode = normalize_answer_mode(answer_mode)


    with measure_stage("pipeline.retrieval"):
        retrieval_result = retrieve_all(
            query=question,
            source_mode=source_mode,
            answer_mode=answer_mode,
        )

    with measure_stage("pipeline.selection"):
        selection = select_context_items(
            retrieval_result=retrieval_result,
            primary_k=primary_k,
            secondary_k=secondary_k,
            tertiary_k=tertiary_k,
            use_combined_fallback=use_combined_fallback,
        )

    with measure_stage("pipeline.context_formatting"):
        context_package = prepare_context_package(
            selection["selected_items"]
        )

    with measure_stage("pipeline.answer_generation"):
        answer_result = generate_final_answer(
            question=question,
            context_blocks=context_package["context_blocks"],
            citation_candidates=context_package["citation_candidates"],
            selection=selection,
            max_citations=max_citations,
        )

    return {
        "question": question,
        "source_mode": source_mode,
        "answer_mode": answer_mode,
        "retrieval_result": retrieval_result,
        "selection": selection,
        "context_package": context_package,
        "answer_result": answer_result,
    }

@traceable(
    name="answer_question",
    run_type="chain",
    tags=["cbr-llp-copilot", "streamlit", "rag", "demo"],
)
def answer_question(
    question: str,
    source_mode: str = DEFAULT_SOURCE_MODE,
    answer_mode: str = DEFAULT_ANSWER_MODE,
    primary_k: int = 3,
    secondary_k: int = 2,
    tertiary_k: int = 0,
    use_combined_fallback: bool = True,
    max_citations: int = 5,
) -> AnswerResult:
    """
    Упрощённая внешняя точка входа.
    Возвращает только финальный AnswerResult.

    Используется Streamlit UI.
    """
    pipeline_result = run_answer_pipeline(
        question=question,
        source_mode=source_mode,
        answer_mode=answer_mode,
        primary_k=primary_k,
        secondary_k=secondary_k,
        tertiary_k=tertiary_k,
        use_combined_fallback=use_combined_fallback,
        max_citations=max_citations,
    )

    return pipeline_result["answer_result"]