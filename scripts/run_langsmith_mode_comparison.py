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


MODE_CONFIGS = [
    {
        "source_mode": "primary",
        "answer_mode": "norm_plus_explanation",
        "experiment_prefix": "cbr-llp-copilot-retrieval-primary",
    },
    {
        "source_mode": "secondary",
        "answer_mode": "norm_plus_explanation",
        "experiment_prefix": "cbr-llp-copilot-retrieval-secondary",
    },
]


def normalize_doc_id(doc_id: str) -> str:
    return DOC_ID_ALIASES.get(doc_id, doc_id)


def normalize_clause(clause: str) -> str:
    if not clause:
        return ""

    clause = str(clause).lower().replace(",", ".")

    matches = re.findall(
        r"\b\d+(?:\.\d+)+\b",
        clause,
    )

    for match in matches:
        if re.fullmatch(
            r"\d{1,2}\.\d{1,2}\.\d{4}",
            match,
        ):
            continue

        return match

    return ""


def parse_metadata_clauses(value) -> set[str]:
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


def extract_clauses_from_text(
    text: str,
) -> set[str]:
    if not text:
        return set()

    matches = re.findall(
        r"\b\d+(?:\.\d+){1,3}\b",
        text,
    )

    result = set()

    for match in matches:
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
    result = set()

    for item in retrieval_result.get(
        "combined_results",
        [],
    ):
        source_doc = item.get("source_doc")

        if source_doc:
            result.add(source_doc)

    return result


def extract_retrieved_clauses(
    retrieval_result: dict,
) -> set[str]:
    result = set()

    for item in retrieval_result.get(
        "combined_results",
        [],
    ):
        document = item.get("document")

        if document is None:
            continue

        metadata = document.metadata or {}

        metadata_clauses = (
            parse_metadata_clauses(
                metadata.get("clauses")
            )
        )

        result.update(metadata_clauses)

        if not metadata_clauses:
            result.update(
                extract_clauses_from_text(
                    document.page_content or ""
                )
            )

    return result


def make_target(
    source_mode: str,
    answer_mode: str,
):
    """
    Создает target-функцию для конкретного
    режима retrieval.
    """

    def target(inputs: dict) -> dict:
        question = inputs["question"]

        retrieval_result = retrieve_all(
            query=question,
            source_mode=source_mode,
            answer_mode=answer_mode,
        )

        return {
            "case_id": inputs.get("case_id"),
            "source_mode": source_mode,
            "answer_mode": answer_mode,
            "effective_source_ids": (
                retrieval_result.get(
                    "source_ids",
                    [],
                )
            ),
            "retrieved_doc_ids": sorted(
                extract_retrieved_doc_ids(
                    retrieval_result
                )
            ),
            "retrieved_clauses": sorted(
                extract_retrieved_clauses(
                    retrieval_result
                )
            ),
        }

    return target


def retrieval_doc_recall(
    inputs: dict,
    outputs: dict,
    reference_outputs: dict,
) -> dict:

    expected_doc_ids = {
        normalize_doc_id(doc_id)
        for doc_id
        in reference_outputs.get(
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
        score = (
            len(
                expected_doc_ids
                & retrieved_doc_ids
            )
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

    expected_clauses = {
        normalize_clause(clause)
        for clause
        in reference_outputs.get(
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
        score = (
            len(
                expected_clauses
                & retrieved_clauses
            )
            / len(expected_clauses)
        )

    return {
        "key": "retrieval_clause_recall",
        "score": score,
    }


def main():
    client = Client()

    for config in MODE_CONFIGS:
        source_mode = config["source_mode"]
        answer_mode = config["answer_mode"]
        experiment_prefix = (
            config["experiment_prefix"]
        )

        print()
        print("==============================")
        print(
            f"Running source_mode="
            f"{source_mode}"
        )
        print(
            f"answer_mode="
            f"{answer_mode}"
        )
        print("==============================")
        print()

        target = make_target(
            source_mode=source_mode,
            answer_mode=answer_mode,
        )

        results = client.evaluate(
            target,
            data=DATASET_NAME,
            evaluators=[
                retrieval_doc_recall,
                retrieval_clause_recall,
            ],
            experiment_prefix=(
                experiment_prefix
            ),
            max_concurrency=1,
        )

        print()
        print(
            f"Completed: {source_mode}"
        )
        print(results)


if __name__ == "__main__":
    main()