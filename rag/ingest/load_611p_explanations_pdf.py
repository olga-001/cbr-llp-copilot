from pathlib import Path
import re

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DOCS_DIR = BASE_DIR / "data" / "raw_docs"
PDF_PATH = RAW_DOCS_DIR / "611P_explanations.pdf"


def normalize_text(text: str) -> str:
    if not text:
        return ""

    # приводим переносы строк к одному виду
    text = text.replace("\r", "\n")

    # убираем повторяющиеся пробелы
    text = re.sub(r"[ \t]+", " ", text)

    # убираем слишком длинные цепочки пустых строк
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def extract_full_text_from_pdf() -> str:
    loader = PyPDFLoader(str(PDF_PATH))
    pages = loader.load()

    texts = []
    for page in pages:
        page_text = normalize_text(page.page_content)
        if page_text:
            texts.append(page_text)

    return "\n\n".join(texts)


def split_into_qa_blocks(full_text: str) -> list[str]:
    """
    Делим весь текст на блоки по заголовкам 'Вопрос'.
    Внутри каждого блока ожидаем:
    - тема / заголовок
    - 'Вопрос'
    - дата вопроса
    - текст вопроса
    - 'Ответ'
    - дата ответа
    - текст ответа
    """
    # разделяем по слову "Вопрос", но сохраняем его в блоке
    parts = re.split(r"\n(?=Вопрос\s*\n)", full_text)

    blocks = []
    current_topic = None

    for part in parts:
        part = part.strip()
        if not part:
            continue

        if "Вопрос" in part and "Ответ" in part:
            blocks.append(part)

    return blocks


def parse_qa_block(block: str, block_id: int) -> Document:
    lines = [line.strip() for line in block.split("\n") if line.strip()]

    # Ищем позиции "Вопрос" и "Ответ"
    question_idx = None
    answer_idx = None

    for i, line in enumerate(lines):
        if line == "Вопрос" and question_idx is None:
            question_idx = i
        elif line == "Ответ" and answer_idx is None:
            answer_idx = i

    if question_idx is None or answer_idx is None:
        raise ValueError(f"Не удалось распарсить блок Q&A #{block_id}")

    # Всё до "Вопрос" считаем темой
    topic_lines = lines[:question_idx]
    topic = " ".join(topic_lines).strip()

    # После "Вопрос" обычно дата, потом текст вопроса
    question_date = ""
    answer_date = ""

    question_text_start = question_idx + 1
    if question_text_start < len(lines) and re.match(r"от \d{2}\.\d{2}\.\d{4}", lines[question_text_start]):
        question_date = lines[question_text_start]
        question_text_start += 1

    question_text = " ".join(lines[question_text_start:answer_idx]).strip()

    # После "Ответ" обычно дата, потом текст ответа
    answer_text_start = answer_idx + 1
    if answer_text_start < len(lines) and re.match(r"от \d{2}\.\d{2}\.\d{4}", lines[answer_text_start]):
        answer_date = lines[answer_text_start]
        answer_text_start += 1

    answer_text = " ".join(lines[answer_text_start:]).strip()

    content = (
        f"Источник: Разъяснения по Положению Банка России №611-П.\n"
        f"Тема: {topic}\n"
        f"Дата вопроса: {question_date}\n"
        f"Дата ответа: {answer_date}\n\n"
        f"Вопрос: {question_text}\n\n"
        f"Ответ: {answer_text}"
    )

    metadata = {
        "doc_id": "611P_EXPL",
        "doc_title": "Разъяснения по 611-П",
        "source_file": PDF_PATH.name,
        "topic": topic,
        "question_date": question_date,
        "answer_date": answer_date,
        "block_id": block_id,
    }

    return Document(page_content=content, metadata=metadata)


def load_611p_explanations_pdf() -> list[Document]:
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"Не найден файл: {PDF_PATH}")

    full_text = extract_full_text_from_pdf()
    blocks = split_into_qa_blocks(full_text)

    docs = []
    for i, block in enumerate(blocks, start=1):
        try:
            doc = parse_qa_block(block, block_id=i)
            docs.append(doc)
        except Exception as e:
            print(f"Ошибка парсинга блока {i}: {e}")

    return docs


def main():
    docs = load_611p_explanations_pdf()
    print("Q&A документов 611-П создано:", len(docs))

    for i, doc in enumerate(docs[:3], start=1):
        print("\n" + "=" * 90)
        print(f"DOCUMENT {i}")
        print("Metadata:", doc.metadata)
        print("-" * 90)
        print(doc.page_content[:1500])


if __name__ == "__main__":
    main()