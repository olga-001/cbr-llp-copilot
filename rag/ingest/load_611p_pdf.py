from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.utils.clean_text import clean_consultant_noise

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DOCS_DIR = BASE_DIR / "data" / "raw_docs"
PDF_PATH = RAW_DOCS_DIR / "611P.pdf"


def load_611p_pdf() -> list[Document]:
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"Не найден файл: {PDF_PATH}")

    loader = PyPDFLoader(str(PDF_PATH))
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200
    )

    docs: list[Document] = []
    for page_num, d in enumerate(pages, start=1):
        # пропускаем титульник
        if page_num == 1:
            continue

        text = clean_consultant_noise(d.page_content)

        if not text or len(text) < 50:
            continue

        chunks = splitter.create_documents([text])

        for chunk_num, cd in enumerate(chunks, start=1):
            md = dict(d.metadata) if d.metadata else {}
            md.update({
                "doc_id": "611P",
                "doc_title": "Положение Банка России №611-П",
                "source_file": PDF_PATH.name,
                "page_num": page_num,
                "chunk_num": chunk_num,
            })

            docs.append(
                Document(
                    page_content="passage: " + cd.page_content,
                    metadata=md
                )
            )

    return docs


def main():
    docs = load_611p_pdf()
    print("Чанков 611-П создано:", len(docs))

    ex = docs[0]
    print("\n--- Пример чанка ---")
    print("Metadata:", ex.metadata)
    print(ex.page_content[:900])


if __name__ == "__main__":
    main()