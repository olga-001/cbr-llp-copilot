import re

from dotenv import load_dotenv

from rag.answer_pipeline import run_answer_pipeline


load_dotenv()


DOC_ID_ALIASES = {
    "590P_EXPLANATIONS": "590P_EXPL",
    "611P_EXPLANATIONS": "611P_EXPL",
}


def normalize_doc_id(doc_id: str) -> str:
    """
    Приводит doc_id из evaluation dataset
    к идентификаторам retrieval-layer.
    """
    return DOC_ID_ALIASES.get(doc_id, doc_id)


def normalize_clause(clause: str) -> str:
    """
    Приводит обозначение нормативного пункта
    к простой форме, например:

    'п. 3.8 Положения № 590-П' -> '3.8'
    'пп. 3.13.3 Положения № 590-П' -> '3.13.3'
    """
    if not clause:
        return ""

    clause = clause.lower().replace(",", ".")

    matches = re.findall(
        r"\b\d+(?:\.\d+)+\b",
        clause,
    )

    if matches:
        return matches[0]

    return clause.strip()


def extract_retrieved_doc_ids(
    retrieval_result: dict,
) -> set[str]:
    """
    Извлекает doc_id из combined retrieval results.
    """
    doc_ids = set()

    for item in retrieval_result.get(
        "combined_results",
        [],
    ):
        source_doc = item.get("source_doc")

        if source_doc:
            doc_ids.add(source_doc)

    return doc_ids


def parse_metadata_clauses(value) -> set[str]:
    """
    Обрабатывает разные форматы metadata['clauses']:

    ['3.7', '3.9']
    '3.8, 3.7, 3.13.3'
    '5.1'
    """
    result = set()

    if not value:
        return result

    if isinstance(value, list):
        raw_values = value

    elif isinstance(value, str):
        raw_values = re.findall(
            r"\b\d+(?:\.\d+)+\b",
            value,
        )

    else:
        raw_values = [str(value)]

    for raw_value in raw_values:
        normalized = normalize_clause(
            str(raw_value)
        )

        if normalized:
            result.add(normalized)

    return result


def extract_clauses_from_text(text: str) -> set[str]:
    """
    Fallback для primary normative chunks.

    Извлекает возможные номера пунктов непосредственно
    из текста, исключая календарные даты.
    """
    if not text:
        return set()

    matches = re.findall(
        r"\b\d+(?:\.\d+){1,3}\b",
        text,
    )

    result = set()

    for match in matches:
        # Исключаем даты вида 11.01.2021
        if re.fullmatch(
            r"\d{1,2}\.\d{1,2}\.\d{4}",
            match,
        ):
            continue

        normalized = normalize_clause(match)

        if normalized:
            result.add(normalized)

    return result


def extract_retrieved_clauses(
    retrieval_result: dict,
) -> set[str]:
    """
    Извлекает нормативные пункты из retrieval.

    Приоритет:
    1. metadata['clauses']
    2. fallback — поиск номера пункта в page_content
    """
    clauses = set()

    for item in retrieval_result.get(
        "combined_results",
        [],
    ):
        document = item.get("document")

        if document is None:
            continue

        metadata = document.metadata or {}

        metadata_clauses = parse_metadata_clauses(
            metadata.get("clauses")
        )

        clauses.update(metadata_clauses)

        # Fallback нужен прежде всего для primary PDF chunks,
        # где поле clauses часто отсутствует или пустое.
        if not metadata_clauses:
            text_clauses = extract_clauses_from_text(
                document.page_content or ""
            )

            clauses.update(text_clauses)

    return clauses


def print_retrieval_metadata(
    retrieval_result: dict,
) -> None:
    """
    Диагностический вывод retrieval metadata.
    """
    print(
        "\nFULL METADATA OF RETRIEVED ITEMS:"
    )

    combined_results = retrieval_result.get(
        "combined_results",
        [],
    )

    if not combined_results:
        print("No combined retrieval results.")
        return

    for i, item in enumerate(
        combined_results,
        start=1,
    ):
        document = item.get("document")

        if document is None:
            continue

        print(f"\n--- ITEM {i} ---")
        print(
            "source_doc:",
            item.get("source_doc"),
        )
        print(
            "source_type:",
            item.get("source_type"),
        )
        print(
            "rank_score:",
            item.get("rank_score"),
        )
        print("metadata:")
        print(document.metadata)


def calculate_doc_recall(
    retrieved_doc_ids: set[str],
    expected_doc_ids: list[str],
) -> float:
    """
    Recall по ожидаемым источникам.
    """
    expected = {
        normalize_doc_id(doc_id)
        for doc_id in expected_doc_ids
    }

    if not expected:
        return 1.0

    found = expected & retrieved_doc_ids

    return len(found) / len(expected)


def calculate_clause_recall(
    retrieved_clauses: set[str],
    expected_clauses: list[str],
) -> float:
    """
    Recall по ожидаемым нормативным пунктам.
    """
    expected = {
        normalize_clause(clause)
        for clause in expected_clauses
        if normalize_clause(clause)
    }

    if not expected:
        return 1.0

    found = expected & retrieved_clauses

    return len(found) / len(expected)


def evaluate_single_case(
    question: str,
    expected_doc_ids: list[str],
    expected_clauses: list[str],
) -> dict:
    """
    Запускает pipeline и считает retrieval metrics
    для одного evaluation case.
    """

    pipeline_result = run_answer_pipeline(
        question=question,
        source_mode="all",
        answer_mode="norm_plus_explanation",
        primary_k=3,
        secondary_k=2,
        tertiary_k=0,
        use_combined_fallback=True,
        max_citations=5,
    )

    retrieval_result = (
        pipeline_result["retrieval_result"]
    )

    print_retrieval_metadata(
        retrieval_result
    )

    retrieved_doc_ids = (
        extract_retrieved_doc_ids(
            retrieval_result
        )
    )

    retrieved_clauses = (
        extract_retrieved_clauses(
            retrieval_result
        )
    )

    doc_recall = calculate_doc_recall(
        retrieved_doc_ids=retrieved_doc_ids,
        expected_doc_ids=expected_doc_ids,
    )

    clause_recall = calculate_clause_recall(
        retrieved_clauses=retrieved_clauses,
        expected_clauses=expected_clauses,
    )

    return {
        "retrieved_doc_ids": sorted(
            retrieved_doc_ids
        ),
        "retrieved_clauses": sorted(
            retrieved_clauses
        ),
        "retrieval_doc_recall": doc_recall,
        "retrieval_clause_recall": clause_recall,
    }


if __name__ == "__main__":
    # Пока проверяем только EVAL_002.

    question = (
        "Как оценивать качество обслуживания долга "
        "до начала выплат по ссуде?"
    )

    expected_doc_ids = [
        "590P"
    ]

    expected_clauses = [
        "п. 3.8 Положения № 590-П"
    ]

    result = evaluate_single_case(
        question=question,
        expected_doc_ids=expected_doc_ids,
        expected_clauses=expected_clauses,
    )

    print("\n==============================")
    print("RETRIEVAL EVALUATION RESULT")
    print("==============================")

    print("\nRetrieved doc IDs:")
    print(
        result["retrieved_doc_ids"]
    )

    print("\nRetrieved clauses:")
    for clause in result[
        "retrieved_clauses"
    ]:
        print(f"- {clause}")

    print(
        "\nretrieval_doc_recall:",
        result["retrieval_doc_recall"],
    )

    print(
        "retrieval_clause_recall:",
        result["retrieval_clause_recall"],
    )