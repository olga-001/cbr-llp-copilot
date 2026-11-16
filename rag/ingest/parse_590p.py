from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

# корень проекта
BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DOCS_DIR = BASE_DIR / "data" / "raw_docs"

PDF_PATH = RAW_DOCS_DIR / "590P.pdf"  # проверь имя файла в папке raw_docs

def main():
    if not PDF_PATH.exists():
        raise SystemExit(f"Не найден файл: {PDF_PATH}\nПроверь имя 590-П в папке data/raw_docs")

    loader = PyPDFLoader(str(PDF_PATH))
    pages = loader.load()  # список Documents, каждый = 1 страница

    print("Файл:", PDF_PATH.name)
    print("Страниц распознано:", len(pages))

    # покажем первые 2 страницы (первые 2000 символов)
    for i in range(min(2, len(pages))):
        text = pages[i].page_content
        print("\n" + "=" * 80)
        print(f"СТРАНИЦА {i+1}")
        print("=" * 80)
        print(text[:2000])

if __name__ == "__main__":
    main()