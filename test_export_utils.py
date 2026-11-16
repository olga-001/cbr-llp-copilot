
from rag.answer_contract import AnswerResult, Citation
from rag.export_utils import (
    prepare_export_data,
    EXPORT_NOTES,
    SYSTEM_NAME,
)


# ---------------------------------------------------------
# Test 1. Standard answer with one citation
# ---------------------------------------------------------

def test_prepare_export_data():
    """Проверка стандартной подготовки данных для экспорта."""

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
        notes=None,
    )

    export_data = prepare_export_data(question, result)

    assert export_data["question"] == question
    assert export_data["answer"] == result.answer

    assert export_data["sources"] == [
        "Положение Банка России № 590-П, пункт 3.9.4"
    ]

    assert export_data["notes"] == EXPORT_NOTES
    assert export_data["system_name"] == SYSTEM_NAME

    print("\n[OK] Тест 1: стандартный ответ успешно обработан.")

    print("\nПодготовленные данные для экспорта:\n")

    for key, value in export_data.items():
        print(f"{key}: {value}")


# ---------------------------------------------------------
# Test 2. Multiple citations and duplicate removal
# ---------------------------------------------------------

def test_multiple_sources():
    """Проверка нескольких источников и удаления дубликатов."""

    result = AnswerResult(
        answer="Тестовый ответ.",
        citations=[
            Citation(
                source="590-P",
                title="Положение Банка России № 590-П",
                locator="пункт 3.8",
                snippet="Фрагмент 1",
                source_type="primary",
            ),
            Citation(
                source="590-P",
                title="Положение Банка России № 590-П",
                locator="пункт 3.8",
                snippet="Фрагмент 2",
                source_type="primary",
            ),
            Citation(
                source="611-P",
                title="Положение Банка России № 611-П",
                locator="пункт 1.7",
                snippet="Фрагмент 3",
                source_type="primary",
            ),
        ],
        grounded=True,
        answer_mode="norm_only",
        notes=None,
    )

    export_data = prepare_export_data(
        question="Тестовый вопрос",
        result=result,
    )

    assert len(export_data["sources"]) == 2

    assert export_data["sources"] == [
        "Положение Банка России № 590-П, пункт 3.8",
        "Положение Банка России № 611-П, пункт 1.7",
    ]

    print(
        "\n[OK] Тест 2: несколько источников "
        "и удаление дубликатов."
    )


# ---------------------------------------------------------
# Test 3. No confirmed citations
# ---------------------------------------------------------

def test_no_sources():
    """Проверка отсутствия подтверждённых источников."""

    question = "Тестовый вопрос без нормативного основания."

    result = AnswerResult(
        answer="Недостаточно нормативных оснований для ответа.",
        citations=[],
        grounded=False,
        answer_mode="insufficient_basis",
        notes=None,
    )

    export_data = prepare_export_data(question, result)

    assert export_data["question"] == question

    assert export_data["answer"] == result.answer

    assert export_data["sources"] == [
        "Подтверждённые источники отсутствуют."
    ]

    assert export_data["notes"] == EXPORT_NOTES
    assert export_data["system_name"] == SYSTEM_NAME

    print(
        "\n[OK] Тест 3: отсутствие подтверждённых "
        "источников обработано корректно."
    )


# ---------------------------------------------------------
# Run all tests
# ---------------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print("EXPORT UTILS TESTS")
    print("=" * 60)

    test_prepare_export_data()
    test_multiple_sources()
    test_no_sources()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
