from rag.ingest.load_590p_pdf import load_590p_pdf

NEEDLES = [
    "3.9.2",
    "использование ссуды заемщиком не по целевому назначению",
    "не по целевому назначению",
]

def main():
    docs = load_590p_pdf()
    print("Чанков загружено:", len(docs))

    found = []
    for d in docs:
        text = (d.page_content or "").lower()
        if any(n.lower() in text for n in NEEDLES):
            found.append(d)

    print("Найдено чанков с совпадениями:", len(found))

    for i, d in enumerate(found[:10], start=1):
        md = d.metadata or {}
        print("\n" + "=" * 90)
        print(f"FOUND {i} | page={md.get('page_num')} chunk={md.get('chunk_num')}")
        # покажем контекст вокруг "не по целев"
        t = d.page_content
        idx = t.lower().find("не по целев")
        if idx == -1:
            idx = t.lower().find("3.9.2")
        start = max(0, idx - 300)
        end = min(len(t), idx + 600)
        print(t[start:end])

if __name__ == "__main__":
    main()