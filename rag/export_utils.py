
from datetime import datetime
from typing import Any, Dict, List

from rag.answer_contract import AnswerResult, Citation


# ---------------------------------------------------------
# Export constants
# ---------------------------------------------------------

SYSTEM_NAME = "CBR Loan Loss Provisions Copilot"

EXPORT_NOTES = (
    "Материал сформирован с использованием AI-ассистента, "
    "носит справочный характер и требует проверки "
    "применимых нормативных положений специалистом."
)


# ---------------------------------------------------------
# Citation formatting
# ---------------------------------------------------------

def format_citation(citation: Citation) -> str:
    """Преобразует источник в текст для справки."""

    title = citation.title.strip()
    locator = citation.locator.strip()

    if title and locator:
        return f"{title}, {locator}"

    return title or locator or citation.source


# ---------------------------------------------------------
# Export data preparation
# ---------------------------------------------------------

def prepare_export_data(
    question: str,
    result: AnswerResult,
) -> Dict[str, Any]:
    """Подготавливает данные для экспорта без вызова LLM."""

    sources: List[str] = []
    seen = set()

    for citation in result.citations:
        formatted = format_citation(citation)

        if formatted and formatted not in seen:
            sources.append(formatted)
            seen.add(formatted)

    if not sources:
        sources = ["Подтверждённые источники отсутствуют."]

    return {
        "question": question,
        "answer": result.answer,
        "sources": sources,
        "notes": EXPORT_NOTES,
        "system_name": SYSTEM_NAME,
    }


# ---------------------------------------------------------
# Copy-ready text formatting
# ---------------------------------------------------------

def format_copy_ready(
    export_data: Dict[str, Any],
) -> str:
    """
    Формирует текстовую версию справки.

    Используется для копирования в буфер обмена.
    Не вызывает LLM и Retrieval.
    """

    created_at = datetime.now().strftime("%d.%m.%Y")

    question = export_data["question"]
    answer = export_data["answer"]
    sources = export_data["sources"]
    notes = export_data["notes"]
    system_name = export_data["system_name"]

    # Format sources
    if len(sources) == 1:
        sources_text = sources[0]
    else:
        sources_text = "\n".join(
            f"{i}. {source}"
            for i, source in enumerate(sources, start=1)
        )

    # Build copy-ready document
    sections = [
        system_name,
        "",
        "Вопрос",
        question,
        "",
        "Ответ",
        answer,
        "",
        "Источник",
        sources_text,
        "",
        "Примечания / ограничения",
        notes,
        "",
        f"Дата формирования: {created_at}",
    ]

    return "\n".join(sections)
