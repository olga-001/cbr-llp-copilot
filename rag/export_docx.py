
from datetime import datetime
from io import BytesIO
from typing import Any, Dict

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor


def add_header_border(paragraph):
    """Добавляет тонкую серую линию под колонтитулом."""

    p_pr = paragraph._p.get_or_add_pPr()

    borders = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")

    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), "4")
    bottom.set(qn("w:space"), "6")
    bottom.set(qn("w:color"), "D9D9D9")

    borders.append(bottom)
    p_pr.append(borders)


def add_section(
    document,
    heading: str,
    content: str,
    italic: bool = False,
):
    """Добавляет раздел справки."""

    heading_paragraph = document.add_paragraph()
    heading_paragraph.paragraph_format.keep_with_next = True
    heading_paragraph.paragraph_format.space_before = Pt(12)
    heading_paragraph.paragraph_format.space_after = Pt(4)

    heading_run = heading_paragraph.add_run(heading)
    heading_run.bold = True
    heading_run.italic = italic

    content_paragraph = document.add_paragraph()
    content_paragraph.paragraph_format.space_after = Pt(0)

    content_run = content_paragraph.add_run(content)
    content_run.italic = italic


def create_docx(
    export_data: Dict[str, Any],
) -> BytesIO:
    """
    Формирует DOCX-справку в оперативной памяти.

    Принимает результат prepare_export_data().
    Не вызывает LLM и Retrieval.
    """

    document = Document()

    # -----------------------------------------------------
    # Page settings
    # -----------------------------------------------------

    section = document.sections[0]

    section.page_width = Cm(21)
    section.page_height = Cm(29.7)

    section.top_margin = Cm(2.5)
    section.bottom_margin = Cm(2.5)
    section.left_margin = Cm(2.5)
    section.right_margin = Cm(2.0)

    section.header_distance = Cm(1.25)
    section.footer_distance = Cm(1.25)

    # -----------------------------------------------------
    # Default text style
    # -----------------------------------------------------

    normal_style = document.styles["Normal"]

    normal_style.font.name = "Times New Roman"
    normal_style.font.size = Pt(12)
    normal_style.font.color.rgb = RGBColor(0, 0, 0)

    normal_style.paragraph_format.line_spacing = 1.15
    normal_style.paragraph_format.space_after = Pt(0)

    # -----------------------------------------------------
    # Header
    # -----------------------------------------------------

    header = section.header
    header_paragraph = header.paragraphs[0]

    header_paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT

    header_run = header_paragraph.add_run(
        export_data["system_name"]
    )

    header_run.font.name = "Times New Roman"
    header_run.font.size = Pt(9)
    header_run.font.color.rgb = RGBColor(100, 100, 100)

    add_header_border(header_paragraph)

    # -----------------------------------------------------
    # Footer
    # -----------------------------------------------------

    footer = section.footer
    footer_paragraph = footer.paragraphs[0]

    footer_paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT

    created_at = datetime.now().strftime("%d.%m.%Y")

    footer_run = footer_paragraph.add_run(created_at)

    footer_run.font.name = "Times New Roman"
    footer_run.font.size = Pt(9)
    footer_run.font.color.rgb = RGBColor(100, 100, 100)

    # -----------------------------------------------------
    # Question
    # -----------------------------------------------------

    add_section(
        document=document,
        heading="Вопрос",
        content=export_data["question"],
    )

    # -----------------------------------------------------
    # Answer
    # -----------------------------------------------------

    add_section(
        document=document,
        heading="Ответ",
        content=export_data["answer"],
    )

    # -----------------------------------------------------
    # Sources
    # -----------------------------------------------------

    sources = export_data["sources"]

    if len(sources) == 1:
        sources_text = sources[0]
    else:
        sources_text = "\n".join(
            f"{i}. {source}"
            for i, source in enumerate(sources, start=1)
        )

    add_section(
        document=document,
        heading="Источник",
        content=sources_text,
    )

    # -----------------------------------------------------
    # Notes and limitations
    # -----------------------------------------------------

    add_section(
        document=document,
        heading="Примечания / ограничения",
        content=export_data["notes"],
        italic=True,
    )

    # -----------------------------------------------------
    # Save DOCX to memory
    # -----------------------------------------------------

    output = BytesIO()

    document.save(output)

    output.seek(0)

    return output
