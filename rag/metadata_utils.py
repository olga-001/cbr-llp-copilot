from copy import deepcopy


SOURCE_TYPE_PRIORITY = {
    "primary": 1,
    "secondary": 2,
    "tertiary": 3,
}


DOC_ID_TO_SOURCE = {
    "590P": "590-P",
    "611P": "611-P",
    "590P_EXPL": "590-P-explanations",
    "611P_EXPL": "611-P-explanations",
}


DOC_ID_TO_TITLE = {
    "590P": "Положение Банка России №590-П",
    "611P": "Положение Банка России №611-П",
    "590P_EXPL": "Разъяснения по 590-П",
    "611P_EXPL": "Разъяснения по 611-П",
}


def normalize_clauses(value):
    """
    Приводит clauses/clause к списку строк.
    """
    if value is None:
        return []

    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]

    if isinstance(value, str):
        return [x.strip() for x in value.split(",") if x.strip()]

    return [str(value).strip()]


def normalize_keywords(value):
    """
    Приводит keywords к списку строк.
    """
    if value is None:
        return []

    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]

    if isinstance(value, str):
        return [x.strip() for x in value.split(",") if x.strip()]

    return [str(value).strip()]


def build_locator(metadata: dict) -> str:
    """
    Строит единое поле locator для citation/UI.
    Приоритет:
    1. locator
    2. question_id
    3. row_id
    4. block_id
    5. clauses
    6. clause
    7. chapter + page
    8. chapter
    9. page
    """
    if metadata.get("locator"):
        return str(metadata["locator"])

    if metadata.get("question_id"):
        return f"вопрос {metadata['question_id']}"

    if metadata.get("row_id") is not None:
        return f"row {metadata['row_id']}"

    if metadata.get("block_id") is not None:
        return f"блок {metadata['block_id']}"

    clauses = metadata.get("clauses")
    if clauses:
        if isinstance(clauses, list):
            return ", ".join(str(x) for x in clauses)
        return str(clauses)

    if metadata.get("clause"):
        return str(metadata["clause"])

    chapter = metadata.get("chapter")
    page = metadata.get("page")

    if chapter and page is not None:
        return f"{chapter}, стр. {page}"
    if chapter:
        return str(chapter)
    if page is not None:
        return f"стр. {page}"

    return "n/a"


def normalize_metadata(metadata: dict) -> dict:
    """
    Нормализует metadata одного Document к единому контракту.

    Возвращает словарь с унифицированными полями:
    - source
    - title
    - source_type
    - locator
    - priority
    - chapter
    - clauses
    - page
    - theme
    - keywords
    - question_id
    - doc_id
    - chunk_id
    - chunk_index
    - row_id
    - block_id
    - question_date
    - answer_date
    """
    md = deepcopy(metadata) if metadata else {}

    doc_id = md.get("doc_id")

    source = md.get("source")
    if not source or (
        isinstance(source, str)
        and (
            source.endswith(".pdf")
            or source.endswith(".xlsx")
            or "\\" in source
            or "/" in source
        )
    ):
        source = DOC_ID_TO_SOURCE.get(doc_id, source or "unknown")

    title = (
        md.get("doc_title")
        or md.get("title")
        or DOC_ID_TO_TITLE.get(doc_id)
        or source
        or "unknown"
    )

    source_type = md.get("source_type", "tertiary")

    priority = md.get("priority")
    if priority is None:
        priority = SOURCE_TYPE_PRIORITY.get(source_type, 99)

    clauses = normalize_clauses(md.get("clauses", md.get("clause")))
    keywords = normalize_keywords(md.get("keywords"))

    locator = build_locator(
        {
            **md,
            "clauses": clauses,
        }
    )

    normalized = {
        "source": source,
        "title": title,
        "source_type": source_type,
        "locator": locator,
        "priority": priority,
        "chapter": md.get("chapter"),
        "clauses": clauses,
        "page": md.get("page"),
        "theme": md.get("theme") or md.get("topic"),
        "keywords": keywords,
        "question_id": md.get("question_id"),
        "doc_id": doc_id,
        "chunk_id": md.get("chunk_id"),
        "chunk_index": md.get("chunk_index", md.get("chunk_num")),
        "row_id": md.get("row_id"),
        "block_id": md.get("block_id"),
        "question_date": md.get("question_date"),
        "answer_date": md.get("answer_date"),
    }

    return normalized


def normalize_retrieval_result_item(item: dict) -> dict:
    """
    Нормализует retrieval result item формата:
    {
        "document": Document(...),
        "rank_score": ...,
        "source_doc": ...,
        "source_type": ...
    }

    Возвращает item того же формата, но с нормализованными metadata у document.
    """
    doc = item["document"]

    base_metadata = doc.metadata or {}
    merged_metadata = {
        **base_metadata,
        "source_type": item.get("source_type", base_metadata.get("source_type")),
    }

    normalized_metadata = normalize_metadata(merged_metadata)
    doc.metadata = normalized_metadata

    return {
        "document": doc,
        "rank_score": item.get("rank_score"),
        "source_doc": item.get("source_doc"),
        "source_type": item.get("source_type"),
    }