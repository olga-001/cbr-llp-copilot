import json
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, ValidationError

from rag.answer_contract import AnswerResult, Citation
from rag.answer_prompt import SYSTEM_PROMPT, build_user_prompt
from rag.llm_config import (
    GGUF_MODEL_PATH,
    LLAMA_MAX_TOKENS,
    LLAMA_N_BATCH,
    LLAMA_N_CTX,
    LLAMA_N_GPU_LAYERS,
    LLAMA_N_THREADS,
    LLAMA_REPEAT_PENALTY,
    LLAMA_TEMPERATURE,
    LLAMA_TOP_P,
    LLAMA_VERBOSE,
)
from langsmith import traceable

try:
    from llama_cpp import Llama
except ImportError as e:
    raise ImportError(
        "Не установлен llama-cpp-python. "
        "Установи пакет: pip install llama-cpp-python "
        "--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu"
    ) from e


class AnswerLLMOutput(BaseModel):
    answer: str
    grounded: bool
    answer_mode: str
    notes: Optional[str] = None


@lru_cache(maxsize=1)
def get_llm() -> Llama:
    return Llama(
        model_path=str(GGUF_MODEL_PATH),
        n_ctx=LLAMA_N_CTX,
        n_threads=LLAMA_N_THREADS,
        n_gpu_layers=LLAMA_N_GPU_LAYERS,
        n_batch=LLAMA_N_BATCH,
        verbose=LLAMA_VERBOSE,
    )


def build_model_prompt(
    question: str,
    context_blocks: str,
    selection: Optional[Dict[str, Any]] = None,
) -> str:
    user_prompt = build_user_prompt(
        question=question,
        context_blocks=context_blocks,
        selection=selection,
    )

    json_instruction = """
Верни результат строго в виде JSON-объекта без markdown, без пояснений до JSON и после JSON.

Обязательная структура:
{
  "answer": "строка",
  "grounded": true,
  "answer_mode": "norm_only" или "norm_plus_explanation" или "insufficient_basis",
  "notes": "строка или null"
}

Требования к answer:
- answer должен быть содержательным и по существу;
- answer не должен быть просто "да", "нет", "не вправе", "вправе";
- answer должен состоять минимум из 2 полных предложений, если в контексте есть достаточная норма;
- если есть первичный источник, сначала сформулируй норму по нему;
- если есть вторичный источник, кратко добавь пояснение только после нормы;
- не используй формулировки, которых нет в переданном контексте.
""".strip()

    return f"{SYSTEM_PROMPT}\n\n{user_prompt}\n\n{json_instruction}\n"


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _extract_json_candidate(text: str) -> str:
    text = _strip_code_fences(text)

    if text.startswith("{") and text.endswith("}"):
        return text

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        return match.group(0)

    raise ValueError("Не удалось извлечь JSON-объект из ответа модели.")


def _repair_json_string(text: str) -> str:
    text = text.strip()

    replacements = {
        "“": '"',
        "”": '"',
        "„": '"',
        "«": '"',
        "»": '"',
        "’": "'",
        "‘": "'",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"\bTrue\b", "true", text)
    text = re.sub(r"\bFalse\b", "false", text)
    text = re.sub(r"\bNone\b", "null", text)

    text = re.sub(r",\s*}", "}", text)
    text = re.sub(r",\s*]", "]", text)

    return text


def _coerce_answer_mode(value: Any) -> str:
    allowed = {"norm_only", "norm_plus_explanation", "insufficient_basis"}
    if isinstance(value, str) and value in allowed:
        return value
    return "insufficient_basis"


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        value = value.strip().lower()
        if value in {"true", "1", "yes"}:
            return True
        if value in {"false", "0", "no"}:
            return False
    return False


def _normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _tokenize(text: str) -> List[str]:
    text = text.lower()
    return re.findall(r"[a-zа-яё0-9\-]{2,}", text, flags=re.IGNORECASE)


def _bigrams(tokens: List[str]) -> Set[str]:
    if len(tokens) < 2:
        return set()
    return {f"{tokens[i]} {tokens[i + 1]}" for i in range(len(tokens) - 1)}


def _is_unacceptably_short_answer(text: str) -> bool:
    if not text:
        return True

    normalized = _normalize_spaces(text).lower().strip(" .,!?:;")

    banned_short = {
        "да",
        "нет",
        "не вправе",
        "вправе",
        "да вправе",
        "нет не вправе",
        "недостаточно оснований",
    }
    if normalized in banned_short:
        return True

    if len(normalized) < 35:
        return True

    sentence_count = len(re.findall(r"[.!?]", text))
    if sentence_count == 0 and len(normalized) < 100:
        return True

    return False


def _selection_based_mode(selection: Optional[Dict[str, Any]]) -> str:
    if not selection:
        return "insufficient_basis"

    has_primary = selection.get("has_primary", False)
    has_secondary = selection.get("has_secondary", False)

    if has_primary and has_secondary:
        return "norm_plus_explanation"
    if has_primary:
        return "norm_only"
    return "insufficient_basis"


def _selection_based_grounded(selection: Optional[Dict[str, Any]]) -> bool:
    if not selection:
        return False
    return bool(selection.get("has_primary", False) or selection.get("has_secondary", False))


def _clean_excerpt(text: str) -> str:
    text = _normalize_spaces(text)
    text = re.sub(r"^passage:\s*", "", text, flags=re.IGNORECASE)
    return text.strip()


def _extract_relevant_excerpt(text: str, question: str, max_len: int = 520) -> str:
    text = _clean_excerpt(text)
    q_tokens = sorted(set(_tokenize(question)), key=len, reverse=True)
    text_lower = text.lower()

    best_pos = None
    for token in q_tokens:
        pos = text_lower.find(token)
        if pos != -1:
            best_pos = pos
            break

    if best_pos is None:
        excerpt = text[:max_len].strip()
        if len(text) > max_len:
            excerpt += "..."
        return excerpt

    start = max(0, best_pos - 180)
    end = min(len(text), best_pos + max_len)
    excerpt = text[start:end].strip()

    if start > 0:
        excerpt = "..." + excerpt
    if end < len(text):
        excerpt += "..."

    return excerpt


def _get_candidate_text(candidate: dict) -> str:
    return candidate.get("full_text") or candidate.get("snippet") or ""


def _metadata_text(candidate: dict) -> str:
    parts = [
        str(candidate.get("title") or ""),
        str(candidate.get("locator") or ""),
        str(candidate.get("source") or ""),
        str(candidate.get("source_type") or ""),
        str(candidate.get("doc_ref") or ""),
    ]

    if candidate.get("theme"):
        parts.append(str(candidate["theme"]))

    if candidate.get("keywords"):
        keywords = candidate["keywords"]
        if isinstance(keywords, list):
            parts.extend(str(x) for x in keywords)
        else:
            parts.append(str(keywords))

    if candidate.get("clauses"):
        clauses = candidate["clauses"]
        if isinstance(clauses, list):
            parts.extend(str(x) for x in clauses)
        else:
            parts.append(str(clauses))

    return " ".join(parts)


def _question_profile(question: str) -> Dict[str, Any]:
    q = _normalize_spaces(question.lower())
    tokens = _tokenize(q)
    token_set = set(tokens)

    stop_words = {
        "как", "какая", "какой", "какие", "каким", "какую",
        "в", "во", "и", "или", "по", "на", "при", "для", "из",
        "ли", "что", "если", "когда", "у", "не", "над", "под",
        "от", "до", "с", "со", "же", "то", "это", "эта", "этой",
        "этот", "эти", "бы", "быть", "который", "которая", "которых",
        "относится", "определяется", "определить", "случае",
    }

    strong_tokens = {
        t for t in token_set
        if len(t) >= 5 and t not in stop_words
    }

    bigrams = _bigrams(tokens)

    has_numeric_or_roman = bool(re.search(r"\b\d+([.,]\d+)?\b", q)) or bool(
        re.search(r"\b(i|ii|iii|iv|v|vi|vii|viii|ix|x)\b", q, flags=re.IGNORECASE)
    )

    normative_markers = [
        "не выше",
        "не ниже",
        "не менее",
        "не более",
        "в течение",
        "более",
        "менее",
        "категор",
        "резерв",
        "процент",
        "классифиц",
        "качества",
        "срок",
        "квартал",
    ]
    marker_hits = [m for m in normative_markers if m in q]

    return {
        "tokens": token_set,
        "strong_tokens": strong_tokens,
        "bigrams": bigrams,
        "has_numeric_or_roman": has_numeric_or_roman,
        "marker_hits": marker_hits,
        "raw": q,
    }


def _count_markers_in_text(markers: List[str], text: str) -> int:
    text_lower = text.lower()
    return sum(1 for m in markers if m in text_lower)


def _has_direct_rule_pattern(text: str) -> bool:
    text_lower = text.lower()

    direct_patterns = [
        r"\bне выше\b",
        r"\bне ниже\b",
        r"\bне менее\b",
        r"\bне более\b",
        r"\bв течение\b",
        r"\bот \d+ до \d+\b",
        r"\b\d+\s*процент",
        r"\bii категория\b",
        r"\biii категория\b",
        r"\biv категория\b",
        r"\bv категория\b",
        r"\bесли\b",
    ]

    return any(re.search(pattern, text_lower) for pattern in direct_patterns)


def _specificity_score(text: str) -> float:
    text_lower = text.lower()

    score = 0.0

    # признаки более "операционной" нормы
    if re.search(r"\b\d+([.,]\d+)?\b", text_lower):
        score += 18
    if re.search(r"\b\d+\s*процент", text_lower):
        score += 22
    if re.search(r"\b(i|ii|iii|iv|v)\s+категор", text_lower):
        score += 18
    if re.search(r"\bв течение\b", text_lower):
        score += 15
    if re.search(r"\bне выше\b", text_lower):
        score += 18
    if re.search(r"\bне менее\b", text_lower):
        score += 16
    if re.search(r"\bесли\b", text_lower):
        score += 10

    # конструкции условий/порогов/ограничений
    pattern_hits = [
        r"\bболее\b",
        r"\bменее\b",
        r"\bот\b.+\bдо\b",
        r"\bклассифицируется\b",
        r"\bформированием резерва\b",
    ]
    for pattern in pattern_hits:
        if re.search(pattern, text_lower):
            score += 8

    return score


def _score_candidate(question: str, candidate: dict) -> float:
    profile = _question_profile(question)

    text = _clean_excerpt(_get_candidate_text(candidate))
    metadata_text = _metadata_text(candidate)

    text_tokens = set(_tokenize(text))
    metadata_tokens = set(_tokenize(metadata_text))

    text_bigrams = _bigrams(_tokenize(text))
    metadata_bigrams = _bigrams(_tokenize(metadata_text))

    rank_score = float(candidate.get("rank_score") or 0)
    source_type = candidate.get("source_type")

    token_overlap = len(profile["tokens"] & text_tokens)
    strong_token_overlap = len(profile["strong_tokens"] & text_tokens)
    metadata_overlap = len(profile["tokens"] & metadata_tokens)

    bigram_overlap = len(profile["bigrams"] & text_bigrams)
    metadata_bigram_overlap = len(profile["bigrams"] & metadata_bigrams)

    marker_hits_in_text = _count_markers_in_text(profile["marker_hits"], text)
    marker_hits_in_metadata = _count_markers_in_text(profile["marker_hits"], metadata_text)

    score = 0.0

    score += token_overlap * 18
    score += strong_token_overlap * 22
    score += bigram_overlap * 35

    score += metadata_overlap * 8
    score += metadata_bigram_overlap * 12

    score += rank_score

    if source_type == "primary":
        score += 35
    elif source_type == "secondary":
        score += 8

    score += marker_hits_in_text * 18
    score += marker_hits_in_metadata * 8

    if profile["has_numeric_or_roman"] and _has_direct_rule_pattern(text):
        score += 35

    text_len = len(text)
    if 250 <= text_len <= 2600:
        score += 8
    elif text_len < 120:
        score -= 12

    if token_overlap == 0:
        score -= 20
    if strong_token_overlap == 0 and bigram_overlap == 0:
        score -= 15

    return score


def _pick_best_general_primary(question: str, citation_candidates: List[dict]) -> Optional[dict]:
    primary_candidates = [c for c in citation_candidates if c.get("source_type") == "primary"]
    if not primary_candidates:
        return None
    return max(primary_candidates, key=lambda c: _score_candidate(question, c))


def _pick_best_specific_primary(
    question: str,
    citation_candidates: List[dict],
    exclude_doc_ref: Optional[str] = None,
) -> Optional[dict]:
    primary_candidates = [c for c in citation_candidates if c.get("source_type") == "primary"]

    if exclude_doc_ref:
        primary_candidates = [c for c in primary_candidates if c.get("doc_ref") != exclude_doc_ref]

    if not primary_candidates:
        return None

    def combined_specific_score(candidate: dict) -> float:
        text = _clean_excerpt(_get_candidate_text(candidate))
        return _score_candidate(question, candidate) + _specificity_score(text) * 3.0

    return max(primary_candidates, key=combined_specific_score)


def _pick_best_secondary_candidate(question: str, citation_candidates: List[dict]) -> Optional[dict]:
    secondary_candidates = [c for c in citation_candidates if c.get("source_type") == "secondary"]
    if not secondary_candidates:
        return None
    return max(secondary_candidates, key=lambda c: _score_candidate(question, c))


def _cleanup_raw_text_for_notes(raw_text: Optional[str]) -> Optional[str]:
    if not raw_text:
        return None

    cleaned = _strip_code_fences(raw_text)
    cleaned = _normalize_spaces(cleaned)

    bad_markers = [
        "обязательная структура",
        "требования к answer",
        "верни результат строго",
        "ответ должен быть",
        "допустимая структура",
    ]
    lower = cleaned.lower()
    if any(marker in lower for marker in bad_markers):
        return None

    return cleaned[:220] if cleaned else None


def _should_include_specific_primary(
    question: str,
    general_primary: Optional[dict],
    specific_primary: Optional[dict],
) -> bool:
    if not general_primary or not specific_primary:
        return False

    if general_primary.get("doc_ref") == specific_primary.get("doc_ref"):
        return False

    general_text = _clean_excerpt(_get_candidate_text(general_primary))
    specific_text = _clean_excerpt(_get_candidate_text(specific_primary))

    general_specificity = _specificity_score(general_text)
    specific_specificity = _specificity_score(specific_text)

    general_score = _score_candidate(question, general_primary)
    specific_score = _score_candidate(question, specific_primary)

    # Включаем второй фрагмент, если он заметно более "операционный"
    if specific_specificity > general_specificity + 10:
        return True

    # Или если он по суммарному качеству почти не уступает общему
    if specific_score >= general_score - 10 and specific_specificity > 0:
        return True

    return False


def _build_fallback_output(
    question: str,
    selection: Optional[Dict[str, Any]],
    citation_candidates: List[dict],
    raw_text: Optional[str] = None,
    reason: Optional[str] = None,
) -> AnswerLLMOutput:
    general_primary = _pick_best_general_primary(question, citation_candidates)
    specific_primary = _pick_best_specific_primary(
        question,
        citation_candidates,
        exclude_doc_ref=general_primary.get("doc_ref") if general_primary else None,
    )
    best_secondary = _pick_best_secondary_candidate(question, citation_candidates)

    selection_mode = _selection_based_mode(selection)
    grounded = _selection_based_grounded(selection)

    if general_primary:
        general_excerpt = _extract_relevant_excerpt(
            text=_get_candidate_text(general_primary),
            question=question,
            max_len=420,
        )

        answer_parts = [
            f"По найденному фрагменту первичного источника следует: {general_excerpt}"
        ]

        if _should_include_specific_primary(question, general_primary, specific_primary):
            specific_excerpt = _extract_relevant_excerpt(
                text=_get_candidate_text(specific_primary),
                question=question,
                max_len=420,
            )
            answer_parts.append(
                f"Дополнительно специальная норма уточняет следующее: {specific_excerpt}"
            )

        if selection_mode == "norm_plus_explanation" and best_secondary:
            answer_parts.append(
                "Дополнительно в контексте имеются разъяснения, которые могут "
                "использоваться как пояснение, но не заменяют саму норму."
            )

        answer = " ".join(answer_parts)

    elif best_secondary:
        secondary_excerpt = _extract_relevant_excerpt(
            text=_get_candidate_text(best_secondary),
            question=question,
            max_len=420,
        )
        answer = (
            f"В найденных фрагментах обнаружено разъяснение: {secondary_excerpt} "
            "Прямую норму в первичном источнике требуется дополнительно проверить."
        )
    else:
        answer = "В найденных фрагментах недостаточно оснований для уверенного ответа."

    notes_parts: List[str] = []
    if reason:
        notes_parts.append(reason)

    raw_preview = _cleanup_raw_text_for_notes(raw_text)
    if raw_preview:
        notes_parts.append(f"raw_model_output={raw_preview}")

    notes = " | ".join(notes_parts) if notes_parts else None

    return AnswerLLMOutput(
        answer=answer,
        grounded=grounded,
        answer_mode=selection_mode if grounded else "insufficient_basis",
        notes=notes,
    )


def _extract_structured_fields_by_regex(text: str) -> Dict[str, Any]:
    cleaned = _strip_code_fences(text)
    data: Dict[str, Any] = {}

    answer_match = re.search(
        r'"answer"\s*:\s*"(.*?)"\s*(?:,\s*"grounded"|,\s*"answer_mode"|,\s*"notes"|})',
        cleaned,
        flags=re.DOTALL,
    )
    if answer_match:
        data["answer"] = answer_match.group(1).replace('\\"', '"').strip()

    grounded_match = re.search(r'"grounded"\s*:\s*(true|false)', cleaned, flags=re.IGNORECASE)
    if grounded_match:
        data["grounded"] = grounded_match.group(1).lower() == "true"

    mode_match = re.search(
        r'"answer_mode"\s*:\s*"(norm_only|norm_plus_explanation|insufficient_basis)"',
        cleaned,
        flags=re.IGNORECASE,
    )
    if mode_match:
        data["answer_mode"] = mode_match.group(1)

    notes_match = re.search(
        r'"notes"\s*:\s*(null|"(.*?)")',
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if notes_match:
        if notes_match.group(1).lower() == "null":
            data["notes"] = None
        else:
            data["notes"] = (notes_match.group(2) or "").replace('\\"', '"').strip()

    return data


def parse_llm_output(
    raw_text: str,
    question: str,
    selection: Optional[Dict[str, Any]],
    citation_candidates: List[dict],
) -> AnswerLLMOutput:
    cleaned_text = _strip_code_fences(raw_text)

    try:
        json_text = _extract_json_candidate(cleaned_text)
        json_text = _repair_json_string(json_text)
        data = json.loads(json_text)

        if "answer" not in data:
            raise ValueError("В ответе модели отсутствует поле 'answer'.")

        parsed = AnswerLLMOutput(
            answer=str(data.get("answer", "")).strip(),
            grounded=_safe_bool(data.get("grounded")),
            answer_mode=_coerce_answer_mode(data.get("answer_mode")),
            notes=data.get("notes"),
        )

        if _is_unacceptably_short_answer(parsed.answer):
            return _build_fallback_output(
                question=question,
                selection=selection,
                citation_candidates=citation_candidates,
                raw_text=raw_text,
                reason="short_or_binary_answer",
            )

        selection_mode = _selection_based_mode(selection)
        selection_grounded = _selection_based_grounded(selection)

        if selection_grounded and not parsed.grounded:
            parsed.grounded = True

        if selection_mode in {"norm_only", "norm_plus_explanation"} and parsed.answer_mode == "insufficient_basis":
            parsed.answer_mode = selection_mode

        parsed.notes = _cleanup_raw_text_for_notes(parsed.notes)

        return parsed

    except (ValueError, json.JSONDecodeError, ValidationError):
        pass

    try:
        salvaged = _extract_structured_fields_by_regex(cleaned_text)
        if "answer" in salvaged:
            parsed = AnswerLLMOutput(
                answer=str(salvaged.get("answer", "")).strip(),
                grounded=_safe_bool(salvaged.get("grounded")),
                answer_mode=_coerce_answer_mode(salvaged.get("answer_mode")),
                notes=salvaged.get("notes"),
            )

            if _is_unacceptably_short_answer(parsed.answer):
                return _build_fallback_output(
                    question=question,
                    selection=selection,
                    citation_candidates=citation_candidates,
                    raw_text=raw_text,
                    reason="regex_salvaged_but_short",
                )

            selection_mode = _selection_based_mode(selection)
            selection_grounded = _selection_based_grounded(selection)

            if selection_grounded and not parsed.grounded:
                parsed.grounded = True

            if selection_mode in {"norm_only", "norm_plus_explanation"} and parsed.answer_mode == "insufficient_basis":
                parsed.answer_mode = selection_mode

            parsed.notes = _cleanup_raw_text_for_notes(parsed.notes)

            return parsed

    except Exception:
        pass

    return _build_fallback_output(
        question=question,
        selection=selection,
        citation_candidates=citation_candidates,
        raw_text=raw_text,
        reason="json_parse_failed",
    )


def _truncate_context_by_chars(context_blocks: str, max_chars: int = 7000) -> str:
    if len(context_blocks) <= max_chars:
        return context_blocks
    return context_blocks[:max_chars]


def generate_answer_structured(
    question: str,
    context_blocks: str,
    citation_candidates: List[dict],
    selection: Optional[Dict[str, Any]] = None,
) -> AnswerLLMOutput:
    llm = get_llm()

    trimmed_context_blocks = _truncate_context_by_chars(
        context_blocks=context_blocks,
        max_chars=7000,
    )

    prompt = build_model_prompt(
        question=question,
        context_blocks=trimmed_context_blocks,
        selection=selection,
    )

    response = llm(
        prompt,
        max_tokens=LLAMA_MAX_TOKENS,
        temperature=LLAMA_TEMPERATURE,
        top_p=LLAMA_TOP_P,
        repeat_penalty=LLAMA_REPEAT_PENALTY,
        stop=["</s>", "<|eot_id|>"],
    )

    raw_text = response["choices"][0]["text"].strip()

    return parse_llm_output(
        raw_text=raw_text,
        question=question,
        selection=selection,
        citation_candidates=citation_candidates,
    )


def merge_llm_output_with_citations(
    llm_output: AnswerLLMOutput,
    citation_candidates: List[dict],
    max_citations: int = 5,
) -> AnswerResult:
    citations: List[Citation] = []

    for candidate in citation_candidates[:max_citations]:
        citations.append(
            Citation(
                source=candidate["source"],
                title=candidate["title"],
                locator=candidate["locator"],
                snippet=candidate["snippet"],
                source_type=candidate["source_type"],
            )
        )

    return AnswerResult(
        answer=llm_output.answer,
        citations=citations,
        grounded=bool(llm_output.grounded),
        answer_mode=_coerce_answer_mode(llm_output.answer_mode),
        notes=llm_output.notes,
    )


# ---------------------------------------------------------
# LangSmith: compact Answer Generation tracing
# ---------------------------------------------------------

def summarize_generation_input(inputs: dict) -> dict:
    """
    Record generation parameters without full context
    or citation candidate texts.
    """
    context = inputs.get("context_blocks") or ""
    candidates = inputs.get("citation_candidates") or []

    return {
        "question": inputs.get("question"),
        "context_length_chars": len(context),
        "citation_candidates_count": len(candidates),
        "max_citations": inputs.get("max_citations", 5),
    }


def summarize_generation_output(result: AnswerResult) -> dict:
    """
    Record key answer metrics without full text.
    The original AnswerResult remains unchanged.
    """
    return {
        "answer_length_chars": len(result.answer or ""),
        "citations_count": len(result.citations or []),
        "grounded": result.grounded,
        "answer_mode": result.answer_mode,
        "has_notes": bool(result.notes),
    }

@traceable(
    name="generate_final_answer",
    run_type="chain",
    tags=["rag", "generation", "local-llm"],
    process_inputs=summarize_generation_input,
    process_outputs=summarize_generation_output,
)
def generate_final_answer(
    question: str,
    context_blocks: str,
    citation_candidates: List[dict],
    selection: Optional[Dict[str, Any]] = None,
    max_citations: int = 5,
) -> AnswerResult:
    llm_output = generate_answer_structured(
        question=question,
        context_blocks=context_blocks,
        citation_candidates=citation_candidates,
        selection=selection,
    )

    return merge_llm_output_with_citations(
        llm_output=llm_output,
        citation_candidates=citation_candidates,
        max_citations=max_citations,
    )