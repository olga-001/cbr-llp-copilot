import argparse
import re
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from langsmith import Client


# Добавляем корень проекта в sys.path,
# чтобы скрипт из папки scripts мог импортировать rag.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from rag.retrieve_hybrid import get_embeddings


load_dotenv()


DEFAULT_EXPERIMENT_NAME = (
    "cbr-llp-copilot-baseline-d0d10cd3"
)

# Рабочий baseline-порог.
# После первого прогона посмотрим распределение similarity
# и при необходимости скорректируем его.
SIMILARITY_THRESHOLD = 0.80


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate final answers for an existing "
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


def cosine_similarity(
    vector_a: list[float],
    vector_b: list[float],
) -> float:
    """
    Cosine similarity двух embedding-векторов.
    """
    a = np.asarray(vector_a, dtype=float)
    b = np.asarray(vector_b, dtype=float)

    denominator = (
        np.linalg.norm(a)
        * np.linalg.norm(b)
    )

    if denominator == 0:
        return 0.0

    return float(
        np.dot(a, b) / denominator
    )


def split_answer_into_chunks(
    answer: str,
) -> list[str]:
    """
    Делит итоговый ответ на небольшие смысловые блоки.

    В список также добавляется полный ответ,
    чтобы не потерять смысл, распределенный
    между несколькими предложениями.
    """
    if not answer:
        return []

    answer = answer.strip()

    chunks = [answer]

    paragraphs = [
        part.strip()
        for part in answer.split("\n")
        if part.strip()
    ]

    chunks.extend(paragraphs)

    sentences = re.split(
        r"(?<=[.!?;])\s+",
        answer,
    )

    chunks.extend(
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    )

    # Убираем дубликаты с сохранением порядка.
    unique_chunks = []
    seen = set()

    for chunk in chunks:
        normalized = chunk.strip()

        if not normalized:
            continue

        if normalized in seen:
            continue

        seen.add(normalized)
        unique_chunks.append(normalized)

    return unique_chunks


def calculate_answer_point_metrics(
    answer: str,
    expected_answer_points: list[str],
    embeddings,
) -> tuple[float, float, str]:
    """
    Для каждого expected_answer_point ищет наиболее
    семантически близкий фрагмент итогового ответа.

    Возвращает:
    - среднюю максимальную similarity;
    - recall по порогу SIMILARITY_THRESHOLD;
    - подробный comment для LangSmith.
    """
    if not expected_answer_points:
        return 1.0, 1.0, "No expected answer points."

    answer_chunks = split_answer_into_chunks(
        answer
    )

    if not answer_chunks:
        return (
            0.0,
            0.0,
            "Answer is empty.",
        )

    # Для multilingual-e5 используем рекомендуемую
    # схему query / passage.
    passage_texts = [
        "passage: " + chunk
        for chunk in answer_chunks
    ]

    passage_vectors = (
        embeddings.embed_documents(
            passage_texts
        )
    )

    point_scores = []

    for expected_point in expected_answer_points:

        query_vector = embeddings.embed_query(
            "query: " + expected_point
        )

        similarities = [
            cosine_similarity(
                query_vector,
                passage_vector,
            )
            for passage_vector
            in passage_vectors
        ]

        best_score = (
            max(similarities)
            if similarities
            else 0.0
        )

        point_scores.append(
            best_score
        )

    average_similarity = (
        sum(point_scores)
        / len(point_scores)
    )

    covered_points = sum(
        score >= SIMILARITY_THRESHOLD
        for score in point_scores
    )

    point_recall = (
        covered_points
        / len(point_scores)
    )

    score_text = ", ".join(
        f"P{i + 1}={score:.3f}"
        for i, score in enumerate(
            point_scores
        )
    )

    comment = (
        f"Threshold={SIMILARITY_THRESHOLD:.2f}; "
        f"{score_text}; "
        f"covered={covered_points}/"
        f"{len(point_scores)}"
    )

    return (
        average_similarity,
        point_recall,
        comment,
    )


def calculate_answer_type_match(
    actual_answer_mode: str,
    expected_answer_type: str,
) -> tuple[float, str]:
    """
    Проверяет совместимость режима ответа
    с ожидаемым типом ответа.

    Для expected 'norm':
    допускается как norm_only,
    так и norm_plus_explanation.

    Для expected 'norm_and_explanation':
    требуется norm_plus_explanation.
    """
    compatibility = {
        "norm": {
            "norm_only",
            "norm_plus_explanation",
        },
        "norm_and_explanation": {
            "norm_plus_explanation",
        },
    }

    allowed_modes = compatibility.get(
        expected_answer_type,
        set(),
    )

    if not allowed_modes:
        score = 0.0

        comment = (
            "Unknown expected answer type: "
            f"{expected_answer_type}"
        )

        return score, comment

    score = (
        1.0
        if actual_answer_mode in allowed_modes
        else 0.0
    )

    comment = (
        f"Expected type: "
        f"{expected_answer_type}; "
        f"actual answer_mode: "
        f"{actual_answer_mode}; "
        f"allowed: "
        f"{sorted(allowed_modes)}"
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

    # Загружается один раз и далее
    # переиспользуется для всех runs.
    embeddings = get_embeddings()

    processed = 0

    for run in runs:

        if not run.reference_example_id:
            print(
                f"Skipping run {run.id}: "
                "no reference_example_id"
            )
            continue

        example = client.read_example(
            example_id=(
                run.reference_example_id
            )
        )

        outputs = run.outputs or {}
        reference_outputs = (
            example.outputs or {}
        )

        case_id = (
            (run.inputs or {}).get(
                "case_id"
            )
            or "unknown"
        )

        answer = outputs.get(
            "answer",
            "",
        )

        actual_answer_mode = (
            outputs.get(
                "answer_mode",
                "",
            )
        )

        expected_answer_points = (
            reference_outputs.get(
                "expected_answer_points",
                [],
            )
        )

        expected_answer_type = (
            reference_outputs.get(
                "expected_answer_type",
                "",
            )
        )

        (
            similarity_score,
            point_recall,
            point_comment,
        ) = calculate_answer_point_metrics(
            answer=answer,
            expected_answer_points=(
                expected_answer_points
            ),
            embeddings=embeddings,
        )

        (
            type_score,
            type_comment,
        ) = calculate_answer_type_match(
            actual_answer_mode=(
                actual_answer_mode
            ),
            expected_answer_type=(
                expected_answer_type
            ),
        )

        client.create_feedback(
            run_id=run.id,
            session_id=run.session_id,
            key=(
                "answer_point_similarity"
            ),
            score=similarity_score,
            comment=point_comment,
        )

        client.create_feedback(
            run_id=run.id,
            session_id=run.session_id,
            key="answer_point_recall",
            score=point_recall,
            comment=point_comment,
        )

        client.create_feedback(
            run_id=run.id,
            session_id=run.session_id,
            key="answer_type_match",
            score=type_score,
            comment=type_comment,
        )

        processed += 1

        print(
            f"{case_id}: "
            f"similarity="
            f"{similarity_score:.3f}, "
            f"point_recall="
            f"{point_recall:.2f}, "
            f"type_match="
            f"{type_score:.2f}"
        )

    print()
    print(
        "Answer feedback uploaded "
        f"for {processed} runs."
    )


if __name__ == "__main__":
    main()