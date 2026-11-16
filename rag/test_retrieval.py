from rag.retrieve_all import retrieve_all


def print_item(prefix: str, idx: int, item: dict):
    doc = item["document"]
    md = doc.metadata or {}

    print("\n" + "=" * 90)
    print(f"{prefix} {idx}")
    print("source_type:", item.get("source_type"))
    print("source_doc:", item.get("source_doc"))
    print("rank_score:", item.get("rank_score"))
    print("page:", md.get("page_num"))
    print("chunk:", md.get("chunk_num"))
    print("row_id:", md.get("row_id"))
    print("block_id:", md.get("block_id"))
    print("clauses:", md.get("clauses"))
    print("theme:", md.get("theme"))
    print("chapter:", md.get("chapter"))
    print("keywords:", md.get("keywords"))
    print("answer_date:", md.get("answer_date"))
    print("-" * 90)
    print(doc.page_content[:1400])


def main():
    query = input("Введите вопрос: ").strip()

    result = retrieve_all(
        query=query,
        top_k_primary=5,
        top_k_secondary=5,
    )

    print("\n" + "#" * 100)
    print("MODE")
    print("#" * 100)
    print(result["mode"])

    print("\n" + "#" * 100)
    print("PRIMARY RESULTS")
    print("#" * 100)
    for i, item in enumerate(result["primary_results"], start=1):
        print_item("PRIMARY", i, item)

    print("\n" + "#" * 100)
    print("SECONDARY RESULTS")
    print("#" * 100)
    for i, item in enumerate(result["secondary_results"], start=1):
        print_item("SECONDARY", i, item)

    print("\n" + "#" * 100)
    print("COMBINED RESULTS")
    print("#" * 100)
    for i, item in enumerate(result["combined_results"], start=1):
        print_item("COMBINED", i, item)


if __name__ == "__main__":
    main()