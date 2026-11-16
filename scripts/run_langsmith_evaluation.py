from dataclasses import asdict, is_dataclass

from dotenv import load_dotenv
from langsmith import Client

from rag.answer_pipeline import answer_question


load_dotenv()

DATASET_NAME = "cbr-llp-copilot-mvp-eval"


def serialize_answer_result(result) -> dict:
    """
    Преобразует AnswerResult в обычный dict,
    который LangSmith сможет сохранить как outputs.
    """

    if is_dataclass(result):
        return asdict(result)

    if hasattr(result, "model_dump"):
        return result.model_dump()

    if hasattr(result, "dict"):
        return result.dict()

    if hasattr(result, "__dict__"):
        return vars(result)

    raise TypeError(
        f"Unsupported AnswerResult type: {type(result)}"
    )


def target(inputs: dict) -> dict:
    """
    Target-функция для LangSmith evaluation.

    LangSmith передает сюда inputs каждого example
    из dataset.
    """

    question = inputs["question"]

    result = answer_question(
        question=question,
        source_mode="all",
        answer_mode="norm_plus_explanation",
        primary_k=3,
        secondary_k=2,
        tertiary_k=0,
        use_combined_fallback=True,
        max_citations=5,
    )

    output = serialize_answer_result(result)

    # Добавляем case_id в actual output,
    # чтобы было удобнее смотреть результаты в LangSmith.
    output["case_id"] = inputs.get("case_id")

    return output


def main():
    client = Client()

    print(f"Dataset: {DATASET_NAME}")
    print("Starting LangSmith experiment...")
    print()

    experiment_results = client.evaluate(
        target,
        data=DATASET_NAME,
        experiment_prefix="cbr-llp-copilot-baseline",
        max_concurrency=1,
    )

    print()
    print("Experiment completed.")
    print(experiment_results)


if __name__ == "__main__":
    main()