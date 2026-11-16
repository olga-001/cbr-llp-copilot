import argparse
import re

from dotenv import load_dotenv
from langsmith import Client


load_dotenv()


DEFAULT_EXPERIMENT_NAME = (
    "cbr-llp-copilot-baseline-d0d10cd3"
)


DOC_ID_ALIASES = {
    "590P_EXPLANATIONS": "590P_EXPL",
    "611P_EXPLANATIONS": "611P_EXPL",

    "590-P": "590P",
    "611-P": "611P",
    "590-P-explanations": "590P_EXPL",
    "611-P-explanations": "611P_EXPL",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate citations for an existing "
            "LangSmith experiment."
        )
    )

    parser.add_argument(
        "experiment_name",
        nargs="?",
        default=DEFAULT_EXPERIMENT_NAME,
        help=(
            "LangSmith experiment name. "
            f"Default: {DEFAULT_EXPERIMENT_NAME}"
        ),
    )

    return parser.parse_args()


def normalize_doc_id(value: str) -> str:
    if not value:
        return ""

    value = str(value).strip()

    return DOC_ID_ALIASES.get(value, value)


def normalize_clause(value: str) -> str:
    if not value:
        return ""

    value = str(value).lower().replace(",", ".")

    matches = re.findall(
        r"\b\d+(?:\.\d+)+\b",
        value,
    )

    for match in matches:
        if re.fullmatch(
            r"\d{1,2}\.\d{1,2}\.\d{4}",
            match,
        ):
            continue

        return match

    return ""


def extract_clauses_from_text(text: str) -> set[str]:
    if not text:
        return set()

    matches = re.findall(
        r"\b\d+(?:\.\d+){1,3}\b",
        str(text),
    )

    result = set()

    for match in matches:
        if re.fullmatch(
            r"\d{1,2}\.\d{1,2}\.\d{4}",
            match,
        ):
            continue

        result.add(match)

    return result


def get_citations(outputs: dict) -> list[dict]:
    if not outputs:
        return []

    citations = outputs.get("citations", [])

    if not isinstance(citations, list):
        return []

    return [
        citation
        for citation in citations
        if isinstance(citation, dict)
    ]


def extract_citation_doc_ids(
    outputs: dict,
) -> set[str]:
    result = set()

    for citation in get_citations(outputs):
        source = normalize_doc_id(
            citation.get("source", "")
        )

        if source:
            result.add(source)

    return result


def extract_citation_clauses(
    outputs: dict,
) -> set[str]:
    result = set()

    for citation in get_citations(outputs):
        locator = citation.get("locator", "")
        snippet = citation.get("snippet", "")

        result.update(
            extract_clauses_from_text(locator)
        )

        result.update(
            extract_clauses_from_text(snippet)
        )

    return result


def calculate_citation_doc_recall(
    outputs: dict,
    reference_outputs: dict,
) -> tuple[float, str]:

    expected_doc_ids = {
        normalize_doc_id(doc_id)
        for doc_id in reference_outputs.get(
            "expected_doc_ids",
            [],
        )
    }

    cited_doc_ids = extract_citation_doc_ids(
        outputs
    )

    if not expected_doc_ids:
        score = 1.0
    else:
        found = (
            expected_doc_ids
            & cited_doc_ids
        )

        score = (
            len(found)
            / len(expected_doc_ids)
        )

    comment = (
        f"Expected: {sorted(expected_doc_ids)}; "
        f"Cited: {sorted(cited_doc_ids)}"
    )

    return score, comment


def calculate_citation_clause_recall(
    outputs: dict,
    reference_outputs: dict,
) -> tuple[float, str]:

    expected_clauses = {
        normalize_clause(clause)
        for clause in reference_outputs.get(
            "expected_clauses",
            [],
        )
        if normalize_clause(clause)
    }

    cited_clauses = extract_citation_clauses(
        outputs
    )

    if not expected_clauses:
        score = 1.0
    else:
        found = (
            expected_clauses
            & cited_clauses
        )

        score = (
            len(found)
            / len(expected_clauses)
        )

    comment = (
        f"Expected: {sorted(expected_clauses)}; "
        f"Found in citations: {sorted(cited_clauses)}"
    )

    return score, comment


def main():
    args = parse_args()

    experiment_name = args.experiment_name

    client = Client()

    print(
        f"Loading runs from experiment: "
        f"{experiment_name}"
    )

    runs = list(
        client.list_runs(
            project_name=experiment_name,
            is_root=True,
        )
    )

    print(f"Runs found: {len(runs)}")
    print()

    processed = 0

    for run in runs:
        if not run.reference_example_id:
            print(
                f"Skipping run {run.id}: "
                f"no reference_example_id"
            )
            continue

        example = client.read_example(
            example_id=run.reference_example_id
        )

        outputs = run.outputs or {}
        reference_outputs = example.outputs or {}

        case_id = (
            (run.inputs or {}).get("case_id")
            or "unknown"
        )

        doc_score, doc_comment = (
            calculate_citation_doc_recall(
                outputs,
                reference_outputs,
            )
        )

        clause_score, clause_comment = (
            calculate_citation_clause_recall(
                outputs,
                reference_outputs,
            )
        )

        client.create_feedback(
            run_id=run.id,
            session_id=run.session_id,
            key="citation_doc_recall",
            score=doc_score,
            comment=doc_comment,
        )

        client.create_feedback(
            run_id=run.id,
            session_id=run.session_id,
            key="citation_clause_recall",
            score=clause_score,
            comment=clause_comment,
        )

        processed += 1

        print(
            f"{case_id}: "
            f"citation_doc_recall={doc_score:.2f}, "
            f"citation_clause_recall={clause_score:.2f}"
        )

    print()
    print(
        f"Citation feedback uploaded "
        f"for {processed} runs."
    )


if __name__ == "__main__":
    main()