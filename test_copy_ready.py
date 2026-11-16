
from rag.answer_contract import AnswerResult, Citation
from rag.export_utils import (
    prepare_export_data,
    format_copy_ready,
    EXPORT_NOTES,
    SYSTEM_NAME,
)


def test_copy_ready():
    """Проверка формирования copy-ready справки."""

    question = (
        "Какая категория качества присваивается ссуде, "
        "если финансовое положение среднее, "
        "а обслуживание долга среднее?"
    )

    result = AnswerResult(
        answer="Ссуда классифицируется в III категорию качества.",
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

    export_data = prepare_export_data(question, result)

    copy_ready_text = format_copy_ready(export_data)

    # Check mandatory sections
    assert "Вопрос" in copy_ready_text
    assert "Ответ" in copy_ready_text
    assert "Источник" in copy_ready_text
    assert "Примечания / ограничения" in copy_ready_text

    # Check content
    assert question in copy_ready_text
    assert result.answer in copy_ready_text
    assert "Положение Банка России № 590-П, пункт 3.9.4" in copy_ready_text

    # Check constants
    assert SYSTEM_NAME in copy_ready_text
    assert EXPORT_NOTES in copy_ready_text

    # Check date
    assert "Дата формирования:" in copy_ready_text

    print("=" * 60)
    print("COPY-READY TEST")
    print("=" * 60)

    print("\n[OK] Все проверки успешно пройдены.")

    print("\nСформированный текст справки:\n")
    print("-" * 60)
    print(copy_ready_text)
    print("-" * 60)


if __name__ == "__main__":
    test_copy_ready()
