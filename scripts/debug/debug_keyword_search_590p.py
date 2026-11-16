from rag.ingest.load_590p_pdf import load_590p_pdf


KEYWORDS = [
    "не по целевому назначению",
    "использование ссуды заемщиком",
    "3.9.2",
]


def score_text(text: str, keywords: list[str]) -> int:
    text_lower = text.lower()
    score = 0
    for kw in keywords:
        if kw.lower() in text_lower:
            score += 1
    return score


def main():
    docs = load_590p_pdf()
    print("Чанков загружено:", len(docs))

    scored = []
    for doc in docs:
        score = score_text(doc.page_content, KEYWORDS)
        if score > 0:
            scored.append((score, doc))

    # сортируем: сначала по score, потом по page_num
    scored.sort(key=lambda x: (x[0], x[1].metadata.get("page_num", 0)), reverse=True)

    print("Найдено чанков с совпадениями:", len(scored))

    top_n = 10
    for i, (score, doc) in enumerate(scored[:top_n], start=1):
        md = doc.metadata or {}
        print("\n" + "=" * 90)
        print(f"РЕЗУЛЬТАТ {i}")
        print("score:", score)
        print("page:", md.get("page_num"))
        print("chunk:", md.get("chunk_num"))
        print("-" * 90)
        print(doc.page_content[:1200])


if __name__ == "__main__":
    main()