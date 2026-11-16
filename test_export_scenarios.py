
from pathlib import Path

from docx import Document

from rag.answer_contract import AnswerResult, Citation
from rag.export_utils import (
    prepare_export_data,
    format_copy_ready,
    EXPORT_NOTES,
    SYSTEM_NAME,
)
from rag.export_docx import create_docx


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def make_citation(
    source: str,
    title: str,
    locator: str,
) -> Citation:
    """Создаёт тестовый нормативный источник."""

    return Citation(
        source=source,
        title=title,
        locator=locator,
        snippet="Тестовый фрагмент нормативного документа",
        source_type="primary",
    )


def generate_document(
    question: str,
    result: AnswerResult,
    filename: str,
):
    """Формирует DOCX и возвращает данные для проверки."""

    export_data = prepare_export_data(question, result)

    copy_text = format_copy_ready(export_data)

    docx_buffer = create_docx(export_data)

    output_path = Path(filename)
    output_path.write_bytes(docx_buffer.getvalue())

    docx_buffer.seek(0)
    document = Document(docx_buffer)

    body_text = "\n".join(
        paragraph.text
        for paragraph in document.paragraphs
    )

    header_text = "\n".join(
        paragraph.text
        for paragraph in document.sections[0].header.paragraphs
    )

    footer_text = "\n".join(
        paragraph.text
        for paragraph in document.sections[0].footer.paragraphs
    )

    # Common checks
    assert question in body_text
    assert result.answer in body_text
    assert EXPORT_NOTES in body_text
    assert SYSTEM_NAME in header_text
    assert footer_text.strip()

    return (
        export_data,
        copy_text,
        document,
        body_text,
        output_path,
    )


# ---------------------------------------------------------
# Test 1. Insufficient basis
# ---------------------------------------------------------

def test_insufficient_basis():

    question = (
        "Как определяется резерв по ситуации, "
        "для которой отсутствуют подтверждённые "
        "нормативные основания?"
    )

    result = AnswerResult(
        answer=(
            "Недостаточно нормативных оснований "
            "для формирования достоверного ответа."
        ),
        citations=[],
        grounded=False,
        answer_mode="insufficient_basis",
    )

    (
        export_data,
        copy_text,
        document,
        body_text,
        output_path,
    ) = generate_document(
        question,
        result,
        "test_export_no_sources.docx",
    )

    expected_message = (
        "Подтверждённые источники отсутствуют."
    )

    assert export_data["sources"] == [expected_message]
    assert expected_message in body_text
    assert expected_message in copy_text
    assert result.answer in copy_text

    print("[OK] Сценарий 1: отсутствие источников.")
    print(f"     Документ: {output_path.resolve()}")


# ---------------------------------------------------------
# Test 2. Long answer
# ---------------------------------------------------------

def test_long_answer():

    question = (
        "Предоставьте подробное разъяснение "
        "порядка формирования резервов."
    )

    test_paragraph = (
        "При формировании резерва кредитная организация "
        "анализирует финансовое положение заёмщика, "
        "качество обслуживания долга и применимые "
        "нормативные требования. "
        "Данный текст используется исключительно "
        "для проверки переноса длинного ответа в DOCX."
    )

    # Long synthetic answer for multi-page testing
    long_answer = "\n\n".join(
        f"Тестовый абзац {i}. {test_paragraph}"
        for i in range(1, 51)
    )

    result = AnswerResult(
        answer=long_answer,
        citations=[
            make_citation(
                "590-P",
                "Положение Банка России № 590-П",
                "пункт 3.9.4",
            )
        ],
        grounded=True,
        answer_mode="norm_plus_explanation",
    )

    (
        export_data,
        copy_text,
        document,
        body_text,
        output_path,
    ) = generate_document(
        question,
        result,
        "test_export_long_answer.docx",
    )

    assert long_answer in body_text
    assert long_answer in copy_text

    assert "Тестовый абзац 1." in body_text
    assert "Тестовый абзац 50." in body_text

    assert len(body_text) > 10000

    print("[OK] Сценарий 2: длинный ответ.")
    print(f"     Символов в ответе: {len(long_answer)}")
    print(f"     Документ: {output_path.resolve()}")
    print(
        "     ВАЖНО: откройте DOCX и проверьте "
        "переносы страниц и колонтитулы."
    )


# ---------------------------------------------------------
# Test 3. Multiple sources and duplicates
# ---------------------------------------------------------

def test_multiple_sources():

    question = (
        "Какие нормативные требования применяются "
        "при формировании резервов?"
    )

    result = AnswerResult(
        answer=(
            "При формировании резервов необходимо "
            "учитывать применимые нормативные требования "
            "Банка России."
        ),
        citations=[
            make_citation(
                "590-P",
                "Положение Банка России № 590-П",
                "пункт 3.8",
            ),
            make_citation(
                "590-P",
                "Положение Банка России № 590-П",
                "пункт 3.8",
            ),
            make_citation(
                "590-P",
                "Положение Банка России № 590-П",
                "пункт 3.9.4",
            ),
            make_citation(
                "611-P",
                "Положение Банка России № 611-П",
                "пункт 1.7",
            ),
        ],
        grounded=True,
        answer_mode="norm_plus_explanation",
    )

    (
        export_data,
        copy_text,
        document,
        body_text,
        output_path,
    ) = generate_document(
        question,
        result,
        "test_export_multiple_sources.docx",
    )

    # Four citations, but only three unique sources
    assert len(export_data["sources"]) == 3

    expected_sources = [
        "Положение Банка России № 590-П, пункт 3.8",
        "Положение Банка России № 590-П, пункт 3.9.4",
        "Положение Банка России № 611-П, пункт 1.7",
    ]

    assert export_data["sources"] == expected_sources

    for i, source in enumerate(expected_sources, start=1):
        numbered_source = f"{i}. {source}"

        assert numbered_source in body_text
        assert numbered_source in copy_text

        # Each unique source appears exactly once
        assert body_text.count(source) == 1

    print("[OK] Сценарий 3: несколько источников.")
    print("[OK] Дублирующиеся источники удалены.")
    print(f"     Документ: {output_path.resolve()}")


# ---------------------------------------------------------
# Run all scenarios
# ---------------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print("EXPORT SCENARIOS TEST")
    print("=" * 60)

    test_insufficient_basis()
    test_long_answer()
    test_multiple_sources()

    print("\n" + "=" * 60)
    print("ALL EXPORT SCENARIOS PASSED")
    print("=" * 60)
