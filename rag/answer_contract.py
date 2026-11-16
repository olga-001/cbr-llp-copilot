from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class Citation(BaseModel):
    source: str = Field(..., description="Код источника, например 590-P, 611-P, 220-I")
    title: str = Field(..., description="Человекочитаемое название источника")
    locator: str = Field(..., description="Пункт, глава, страница, question_id и т.п.")
    snippet: str = Field(..., description="Короткий фрагмент использованного текста")
    source_type: Literal["primary", "secondary", "tertiary"] = Field(
        ..., description="Тип источника"
    )


class AnswerResult(BaseModel):
    answer: str = Field(..., description="Итоговый текст ответа пользователю")
    citations: List[Citation] = Field(default_factory=list)
    grounded: bool = Field(
        ..., description="Есть ли достаточная опора на retrieved context"
    )
    answer_mode: Literal[
        "norm_only",
        "norm_plus_explanation",
        "insufficient_basis"
    ] = Field(..., description="Режим ответа")
    notes: Optional[str] = Field(
        default=None,
        description="Техническая или поясняющая пометка, если нужна"
    )