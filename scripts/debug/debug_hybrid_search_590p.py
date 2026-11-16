from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from rag.ingest.load_590p_pdf import load_590p_pdf

BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = BASE_DIR / "data" / "chroma"
COLLECTION_NAME = "cbr_llp_copilot"

KEYWORDS = [
    "не по целевому назначению",
    "использование ссуды заемщиком",
    "3.9.2",
]


def keyword_score(text: str, keywords: list[str]) -> int:
    text_lower = text.lower()
    score = 0
    for kw in keywords:
        if kw.lower() in text_lower:
            score += 1
    return score


def semantic_search():
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-base"
    )

    db = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    query = (
        "query: нецелевое использование ссуды, использование ссуды не по целевому назначению, "
        "классификация ссуды, размер резерва, 590-П"
    )

    candidates = db.max_marginal_relevance_search(query, k=20, fetch_k=50)
    return [d for d in candidates if (d.metadata or {}).get("doc_id") == "590P"]


def keyword_search():
    docs = load_590p_pdf()
    scored = []

    for doc in docs:
        score = keyword_score(doc.page_content, KEYWORDS)
        if score > 0:
            scored.append((score, doc))

    scored.sort(key=lambda x: (x[0], x[1].metadata.get("page_num", 0)), reverse=True)
    return [doc for score, doc in scored]


def hybrid_merge(semantic_docs, keyword_docs, top_k=10):
    seen = set()
    merged = []

    # Сначала keyword — потому что по 590-П они сейчас точнее
    for doc in keyword_docs + semantic_docs:
        md = doc.metadata or {}
        key = (md.get("page_num"), md.get("chunk_num"))
        if key not in seen:
            seen.add(key)
            merged.append(doc)
        if len(merged) >= top_k:
            break

    return merged


def show(title, docs):
    print("\n" + "#" * 90)
    print(title)
    print("Найдено:", len(docs))

    for i, doc in enumerate(docs, start=1):
        md = doc.metadata or {}
        print("\n" + "=" * 80)
        print(f"РЕЗУЛЬТАТ {i}")
        print("page:", md.get("page_num"))
        print("chunk:", md.get("chunk_num"))
        print("-" * 80)
        print(doc.page_content[:1200])


def main():
    semantic_docs = semantic_search()
    keyword_docs = keyword_search()
    hybrid_docs = hybrid_merge(semantic_docs, keyword_docs, top_k=10)

    show("1) Semantic search", semantic_docs[:5])
    show("2) Keyword search", keyword_docs[:5])
    show("3) Hybrid search", hybrid_docs[:10])


if __name__ == "__main__":
    main()