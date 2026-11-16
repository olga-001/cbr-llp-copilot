from pprint import pprint

from rag.retrieve_all import retrieve_all
from rag.selection_utils import select_context_items


def main():
    question = "Как классифицируются элементы расчетной базы резерва, относящиеся к контрагенту, у которого перед кредитной организацией имеется задолженность по ссуде, ссудная и приравненная к ней задолженность?"

    result = retrieve_all(question)

    selection = select_context_items(
        result,
        primary_k=3,
        secondary_k=2,
        tertiary_k=0,
    )

    print("=" * 100)
    print("SELECTION MODE:")
    print(selection["selection_mode"])

    print("=" * 100)
    print("HAS PRIMARY / HAS SECONDARY:")
    print(selection["has_primary"], selection["has_secondary"])

    print("=" * 100)
    print("SELECTED PRIMARY:")
    for i, item in enumerate(selection["selected_primary"], start=1):
        print(f"\nPRIMARY {i}")
        print("rank_score:", item.get("rank_score"))
        print("metadata:")
        pprint(item["document"].metadata)
        print("content preview:")
        print(item["document"].page_content[:500])

    print("=" * 100)
    print("SELECTED SECONDARY:")
    for i, item in enumerate(selection["selected_secondary"], start=1):
        print(f"\nSECONDARY {i}")
        print("rank_score:", item.get("rank_score"))
        print("metadata:")
        pprint(item["document"].metadata)
        print("content preview:")
        print(item["document"].page_content[:500])

    print("=" * 100)
    print("FINAL SELECTED ITEMS:")
    for i, item in enumerate(selection["selected_items"], start=1):
        print(f"\nITEM {i}")
        print("rank_score:", item.get("rank_score"))
        print("source_type:", item.get("source_type"))
        print("metadata:")
        pprint(item["document"].metadata)
        print("content preview:")
        print(item["document"].page_content[:500])


if __name__ == "__main__":
    main()