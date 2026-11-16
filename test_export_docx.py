
from pathlib import Path
from tempfile import TemporaryDirectory

from docx import Document

from rag.answer_contract import AnswerResult, Citation
from rag.export_utils import (
    prepare_export_data,
    EXPORT_NOTES,
    SYSTEM_NAME,
)
from rag.export_docx import create_docx


def test_export_docx():

    question = (
        "Какая категория качества присваивается ссуде, "
        "если финансовое положение заёмщика оценивается "
        "как среднее, а обслуживание долга — среднее?"
    )

    result = AnswerResult(
        answer=(
            "Ссуда классифицируется в III категорию "
            "качества согласно пункту 3.9.4 "
            "Положения Банка России № 590-П."
        ),
        citations=[
            Citation(
                source="590-P",
                title="Положение Банка России № 590-П",
                locator="пункт 3.9.4",
                snippet="Фрагмент нормативного документа",
                source_type="primary",
            )
        ],
        grounded=True,
        answer_mode="norm_only",
    )

    # Step 1. Prepare data
    export_data = prepare_export_data(
        question=question,
        result=result,
    )

    # Step 2. Generate DOCX
    docx_buffer = create_docx(export_data)

    assert docx_buffer.getbuffer().nbytes > 0

    # Step 3. Reopen DOCX and check content
    document = Document(docx_buffer)

    paragraphs = [
        paragraph.text
        for paragraph in document.paragraphs
    ]

    full_text = "\n".join(paragraphs)

    assert "Вопрос" in full_text
    assert "Ответ" in full_text
    assert "Источник" in full_text
    assert "Примечания / ограничения" in full_text

    assert question in full_text
    assert result.answer in full_text
    assert EXPORT_NOTES in full_text

    assert (
        "Положение Банка России № 590-П, пункт 3.9.4"
        in full_text
    )

    # Check header
    header_text = "\n".join(
        p.text for p in document.sections[0].header.paragraphs
    )

    assert SYSTEM_NAME in header_text

    # Check footer
    footer_text = "\n".join(
        p.text for p in document.sections[0].footer.paragraphs
    )

    assert footer_text.strip()

    # Save test document for visual inspection
    output_path = Path("test_export.docx")

    output_path.write_bytes(docx_buffer.getvalue())

    print("=" * 60)
    print("DOCX EXPORT TEST")
    print("=" * 60)

    print("\n[OK] DOCX успешно сформирован.")
    print("[OK] Все обязательные разделы присутствуют.")
    print("[OK] Содержимое ответа сохранено.")
    print("[OK] Источники сохранены.")
    print("[OK] Примечание присутствует.")
    print("[OK] Верхний и нижний колонтитулы присутствуют.")

    print(f"\nТестовый документ: {output_path.resolve()}")

    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    test_export_docx()
