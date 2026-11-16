from pprint import pprint

from rag.metadata_utils import normalize_metadata


def main():
    test_cases = [
        {
            "name": "590-P primary chunk",
            "metadata": {
                "source": "590-P",
                "title": "Положение Банка России 590-П",
                "source_type": "primary",
                "clause": "3.1",
                "chapter": "Глава 3",
                "page": 17,
                "keywords": ["резерв", "категория качества"],
            },
        },
        {
            "name": "590-P explanations",
            "metadata": {
                "source": "590-P-explanations",
                "title": "Разъяснения по 590-П",
                "source_type": "secondary",
                "question_id": "12",
                "chapter": "Глава 4",
                "clauses": ["4.1", "4.2"],
                "theme": "реструктуризация",
            },
        },
        {
            "name": "611-P PDF chunk with minimum fields",
            "metadata": {
                "source": "611-P",
                "source_type": "primary",
                "page": 8,
            },
        },
        {
            "name": "CBR review / tertiary",
            "metadata": {
                "source": "cbr-review",
                "title": "Обзор банковского регулирования",
                "source_type": "tertiary",
                "block_id": "2",
                "theme": "изменения регулирования",
            },
        },
        {
            "name": "Empty metadata",
            "metadata": {},
        },
    ]

    for case in test_cases:
        print("=" * 80)
        print(case["name"])
        normalized = normalize_metadata(case["metadata"])
        pprint(normalized)
        print()

if __name__ == "__main__":
    main()