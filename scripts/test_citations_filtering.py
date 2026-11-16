import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.answer_pipeline import answer_question


def citation_sources(result):
    citations = getattr(result, "citations", None) or []
    return [
        {
            "source": getattr(citation, "source", None),
            "source_type": getattr(citation, "source_type", None),
            "locator": getattr(citation, "locator", None),
        }
        for citation in citations
    ]


def print_check(title, source_mode, answer_mode, question):
    print("=" * 100)
    print(title)
    print("source_mode:", source_mode)
    print("answer_mode:", answer_mode)

    result = answer_question(
        question=question,
        source_mode=source_mode,
        answer_mode=answer_mode,
    )

    print("result.answer_mode:", getattr(result, "answer_mode", None))
    print("grounded:", getattr(result, "grounded", None))
    print("notes:", getattr(result, "notes", None))
    print("citations:")

    for item in citation_sources(result):
        print(" -", item)


def main():
    question_primary = (
        "Какая категория качества, если финансовое положение среднее "
        "и обслуживание долга среднее?"
    )

    question_secondary = (
        "Что говорится в разъяснениях о категории качества ссуды?"
    )

    checks = [
        (
            "Проверка 1: только 590-П",
            "590p",
            "norm_plus_explanation",
            question_primary,
        ),
        (
            "Проверка 2: только 611-П",
            "611p",
            "norm_plus_explanation",
            question_primary,
        ),
        (
            "Проверка 3: только нормативные документы",
            "primary",
            "norm_plus_explanation",
            question_primary,
        ),
        (
            "Проверка 4: только разъяснения",
            "secondary",
            "norm_plus_explanation",
            question_secondary,
        ),
        (
            "Проверка 5: весь корпус + только норма",
            "all",
            "norm_only",
            question_primary,
        ),
    ]

    for title, source_mode, answer_mode, question in checks:
        print_check(
            title=title,
            source_mode=source_mode,
            answer_mode=answer_mode,
            question=question,
        )


if __name__ == "__main__":
    main()