from pprint import pprint

from rag.answer_pipeline import run_answer_pipeline


def main():
    questions = [
        "Как классифицируется ссуда в случае отсутствие финансовой информации по заемщику?",
        "Как классифицируются элементы расчетной базы резерва, относящиеся к контрагенту, у которого перед кредитной организацией имеется задолженность по ссуде, ссудная и приравненная к ней задолженность?",
        "В случае если у заемщика несколько ссуд как определяется величина резерва по вновь выданной ссуде?",
    ]

    for question in questions:
        print("=" * 120)
        print("QUESTION:")
        print(question)

        result = run_answer_pipeline(question=question)

        print("\nSELECTION MODE:")
        print(result["selection"]["selection_mode"])

        print("\nANSWER RESULT:")
        pprint(result["answer_result"].model_dump())

        print("\nCONTEXT PREVIEW:")
        print(result["context_package"]["context_blocks"][:2000])

        print("\nCITATION CANDIDATES PREVIEW:")
        pprint(result["context_package"]["citation_candidates"][:2])
        print()


if __name__ == "__main__":
    main()