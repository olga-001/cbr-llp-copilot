import argparse

from dotenv import load_dotenv
from langsmith import Client


load_dotenv()


DEFAULT_EXPERIMENT_NAME = (
    "cbr-llp-copilot-baseline-d0d10cd3"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate operational metrics for an "
            "existing LangSmith experiment."
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
        inputs = run.inputs or {}
        outputs = run.outputs or {}

        case_id = (
            inputs.get("case_id")
            or "unknown"
        )

        answer_mode = outputs.get(
            "answer_mode",
            "",
        )

        grounded = outputs.get(
            "grounded"
        )

        answer = outputs.get(
            "answer",
            "",
        )

        citations = outputs.get(
            "citations",
            [],
        )

        notes = outputs.get(
            "notes"
        )

        # 1. Fallback
        fallback_triggered = (
            1.0
            if answer_mode == "insufficient_basis"
            else 0.0
        )

        # 2. Ungrounded result
        ungrounded_flag = (
            1.0
            if grounded is False
            else 0.0
        )

        # 3. Technical run error
        technical_error = (
            1.0
            if run.error
            else 0.0
        )

        # 4. Empty final answer
        empty_answer = (
            1.0
            if not str(answer).strip()
            else 0.0
        )

        # 5. Empty citations
        empty_citations = (
            1.0
            if not citations
            else 0.0
        )

        common_comment = (
            f"case_id={case_id}; "
            f"answer_mode={answer_mode}; "
            f"grounded={grounded}; "
            f"notes={notes}; "
            f"run_error={run.error}"
        )

        client.create_feedback(
            run_id=run.id,
            session_id=run.session_id,
            key="fallback_triggered",
            score=fallback_triggered,
            comment=common_comment,
        )

        client.create_feedback(
            run_id=run.id,
            session_id=run.session_id,
            key="ungrounded_flag",
            score=ungrounded_flag,
            comment=common_comment,
        )

        client.create_feedback(
            run_id=run.id,
            session_id=run.session_id,
            key="technical_error",
            score=technical_error,
            comment=common_comment,
        )

        client.create_feedback(
            run_id=run.id,
            session_id=run.session_id,
            key="empty_answer",
            score=empty_answer,
            comment=common_comment,
        )

        client.create_feedback(
            run_id=run.id,
            session_id=run.session_id,
            key="empty_citations",
            score=empty_citations,
            comment=common_comment,
        )

        processed += 1

        print(
            f"{case_id}: "
            f"fallback={fallback_triggered:.0f}, "
            f"ungrounded={ungrounded_flag:.0f}, "
            f"technical_error={technical_error:.0f}, "
            f"empty_answer={empty_answer:.0f}, "
            f"empty_citations={empty_citations:.0f}"
        )

    print()
    print(
        f"Operational feedback uploaded "
        f"for {processed} runs."
    )


if __name__ == "__main__":
    main()