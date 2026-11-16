from pathlib import Path
import pandas as pd

from langchain_core.documents import Document

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DOCS_DIR = BASE_DIR / "data" / "raw_docs"
FILE_PATH = RAW_DOCS_DIR / "590P_explanations.xlsx"
SHEET_NAME = "Разъяснения по 590-П"


def normalize(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def load_590p_explanations() -> list[Document]:
    if not FILE_PATH.exists():
        raise FileNotFoundError(f"Не найден файл: {FILE_PATH}")

    # Читаем лист без заголовков
    df = pd.read_excel(FILE_PATH, sheet_name=SHEET_NAME, header=None)

    # Ищем строку, где есть "Вопрос" и "Ответ"
    header_row_idx = None
    for i in range(min(50, len(df))):
        row_values = [normalize(x) for x in df.iloc[i].tolist()]
        if "Вопрос" in row_values and "Ответ" in row_values:
            header_row_idx = i
            break

    if header_row_idx is None:
        raise ValueError("Не удалось найти строку заголовков с 'Вопрос' и 'Ответ'.")

    # Делаем найденную строку заголовком
    headers = df.iloc[header_row_idx].tolist()
    data = df.iloc[header_row_idx + 1:].copy()
    data.columns = headers

    # Удаляем пустые строки и мусорные колонки
    data = data.dropna(how="all")
    data = data.loc[:, [c for c in data.columns if pd.notna(c)]]
    data = data.dropna(axis=1, how="all")

    # Удаляем строки без вопроса/ответа
    if "Вопрос" not in data.columns or "Ответ" not in data.columns:
        raise ValueError(f"После нормализации не найдены колонки 'Вопрос'/'Ответ'. Колонки: {list(data.columns)}")

    data = data.dropna(subset=["Вопрос", "Ответ"], how="any")

    docs = []

    for i, row in data.iterrows():
        question = normalize(row.get("Вопрос"))
        answer = normalize(row.get("Ответ"))

        if not question or not answer:
            continue

        clauses = normalize(row.get("Пункты"))
        theme = normalize(row.get("Тематика"))
        keywords = normalize(row.get("Ключевые слова"))
        chapter = normalize(row.get("Глава"))

        content = (
            f"Источник: Разъяснения по Положению Банка России №590-П.\n"
            f"Тематика: {theme}\n"
            f"Глава: {chapter}\n"
            f"Пункты: {clauses}\n"
            f"Ключевые слова: {keywords}\n\n"
            f"Вопрос: {question}\n\n"
            f"Ответ: {answer}"
        )

        metadata = {
            "doc_id": "590P_EXPL",
            "doc_title": "Разъяснения по 590-П",
            "source_file": FILE_PATH.name,
            "sheet_name": SHEET_NAME,
            "row_id": int(i),
            "clauses": clauses,
            "theme": theme,
            "chapter": chapter,
            "keywords": keywords,
        }

        docs.append(
            Document(
                page_content=content,
                metadata=metadata
            )
        )

    return docs


def main():
    docs = load_590p_explanations()
    print("Q&A документов 590-П создано:", len(docs))

    for i, doc in enumerate(docs[:3], start=1):
        print("\n" + "=" * 90)
        print(f"DOCUMENT {i}")
        print("Metadata:", doc.metadata)
        print("-" * 90)
        print(doc.page_content[:1500])


if __name__ == "__main__":
    main()