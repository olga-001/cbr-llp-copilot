from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from rag.ingest.load_590p_pdf import load_590p_pdf
from rag.ingest.load_611p_pdf import load_611p_pdf
from rag.ingest.load_590p_explanations import load_590p_explanations
from rag.ingest.load_611p_explanations_pdf import load_611p_explanations_pdf

BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = BASE_DIR / "data" / "chroma"
COLLECTION_NAME = "cbr_llp_copilot"


def build_index():
    # 1. Загружаем документы
    docs_590 = load_590p_pdf()
    docs_611 = load_611p_pdf()
    docs_590_expl = load_590p_explanations()
    docs_611_expl = load_611p_explanations_pdf()

    all_docs = docs_590 + docs_611 + docs_590_expl + docs_611_expl
    print("Всего документов для индекса:", len(all_docs))

    # 2. Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-base"
    )

    # 3. Создаём новую коллекцию
    db = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    # 4. Добавляем документы
    db.add_documents(all_docs)

    print("✅ Индекс создан.")
    print("Папка индекса:", CHROMA_DIR)


if __name__ == "__main__":
    build_index()