from pprint import pprint

from rag.metadata_utils import normalize_retrieval_result_item
from rag.retrieve_all import retrieve_all


def main():
    question = "Как классифицируются элементы расчетной базы резерва, относящиеся к контрагенту, у которого перед кредитной организацией имеется задолженность по ссуде, ссудная и приравненная к ней задолженность?"

    result = retrieve_all(question)
    items = result["combined_results"]

    if not items:
        print("Retrieval ничего не вернул.")
        return

    for i, item in enumerate(items[:5], start=1):
        print("=" * 100)
        print(f"ITEM {i}")

        print("- RAW ITEM KEYS:")
        print(item.keys())

        print("- RAW DOCUMENT METADATA:")
        pprint(item["document"].metadata)

        normalized_item = normalize_retrieval_result_item(item)

        print("- NORMALIZED DOCUMENT METADATA:")
        pprint(normalized_item["document"].metadata)

        print("- RANK SCORE:")
        print(normalized_item["rank_score"])

        print("- CONTENT PREVIEW:")
        print(normalized_item["document"].page_content[:500])
        print()


if __name__ == "__main__":
    main()