import re

from dotenv import load_dotenv
from langsmith import Client

from rag.retrieve_all import retrieve_all


load_dotenv()


DATASET_NAME = "cbr-llp-copilot-mvp-eval"

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
    к простой форме.

    Примеры:
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
    Fallback для primary нормативных chunks.

    Извлекает номера пунктов из текста,
    но исключает календарные даты.
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


def extract_retrieved_clauses(
    retrieval_result: dict,
) -> set[str]:
    """
    Извлекает нормативные пункты из retrieval.

    Приоритет:
    1. metadata['clauses']
    2. fallback — page_content
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

        if not metadata_clauses:
            clauses.update(
                extract_clauses_from_text(
                    document.page_content or ""
                )
            )

    return clauses


def retrieval_target(inputs: dict) -> dict:
    """
    Target для LangSmith.

    Запускает только retrieval-layer,
    без генерации финального ответа.
    """
    question = inputs["question"]

    retrieval_result = retrieve_all(
        query=question,
        source_mode="all",
        answer_mode="norm_plus_explanation",
    )

    retrieved_doc_ids = extract_retrieved_doc_ids(
        retrieval_result
    )

    retrieved_clauses = extract_retrieved_clauses(
        retrieval_result
    )

    return {
        "case_id": inputs.get("case_id"),
        "retrieved_doc_ids": sorted(
            retrieved_doc_ids
        ),
        "retrieved_clauses": sorted(
            retrieved_clauses
        ),
    }


def retrieval_doc_recall(
    inputs: dict,
    outputs: dict,
    reference_outputs: dict,
) -> dict:
    """
    Доля ожидаемых документов,
    которые были найдены retrieval-layer.
    """
    expected_doc_ids = {
        normalize_doc_id(doc_id)
        for doc_id in reference_outputs.get(
            "expected_doc_ids",
            [],
        )
    }

    retrieved_doc_ids = set(
        outputs.get(
            "retrieved_doc_ids",
            [],
        )
    )

    if not expected_doc_ids:
        score = 1.0
    else:
        found = (
            expected_doc_ids
            & retrieved_doc_ids
        )

        score = (
            len(found)
            / len(expected_doc_ids)
        )

    return {
        "key": "retrieval_doc_recall",
        "score": score,
    }


def retrieval_clause_recall(
    inputs: dict,
    outputs: dict,
    reference_outputs: dict,
) -> dict:
    """
    Доля ожидаемых нормативных пунктов,
    которые были найдены retrieval-layer.
    """
    expected_clauses = {
        normalize_clause(clause)
        for clause in reference_outputs.get(
            "expected_clauses",
            [],
        )
        if normalize_clause(clause)
    }

    retrieved_clauses = set(
        outputs.get(
            "retrieved_clauses",
            [],
        )
    )

    if not expected_clauses:
        score = 1.0
    else:
        found = (
            expected_clauses
            & retrieved_clauses
        )

        score = (
            len(found)
            / len(expected_clauses)
        )

    return {
        "key": "retrieval_clause_recall",
        "score": score,
    }


def main():
    client = Client()

    print(f"Dataset: {DATASET_NAME}")
    print(
        "Starting LangSmith retrieval evaluation..."
    )
    print()

    experiment_results = client.evaluate(
        retrieval_target,
        data=DATASET_NAME,
        evaluators=[
            retrieval_doc_recall,
            retrieval_clause_recall,
        ],
        experiment_prefix=(
            "cbr-llp-copilot-retrieval"
        ),
        max_concurrency=1,
    )

    print()
    print(
        "Retrieval evaluation completed."
    )
    print(experiment_results)


if __name__ == "__main__":
    main()