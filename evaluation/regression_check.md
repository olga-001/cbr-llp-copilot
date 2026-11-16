# CBR Loan Loss Provisions Copilot
## Regression Check Procedure

### Purpose

Повторный evaluation-прогон выполняется после существенных
изменений проекта для выявления регрессий относительно baseline.

---

## Reference dataset

Dataset:

`cbr-llp-copilot-mvp-eval`

Количество кейсов:

`10`

Dataset и reference outputs при regression check
не изменяются.

---

## Baseline experiments

Full pipeline baseline:

`cbr-llp-copilot-baseline-d0d10cd3`

Retrieval baseline:

`cbr-llp-copilot-retrieval-69a00985`

---

## Baseline pipeline configuration

- source_mode: `all`
- answer_mode: `norm_plus_explanation`
- primary_k: `3`
- secondary_k: `2`
- tertiary_k: `0`
- use_combined_fallback: `True`
- max_citations: `5`

Для корректного regression comparison эти параметры
не изменяются, если целью теста не является непосредственно
проверка новой конфигурации поиска.

---

## Regression procedure

### 1. Full pipeline run

Запустить:

`python scripts/run_langsmith_evaluation.py`

Новый experiment должен иметь отдельное имя, например:

`cbr-llp-copilot-regression-*`

Не перезаписывать baseline experiment.

### 2. Citation evaluation

Для нового experiment рассчитать:

- citation_doc_recall
- citation_clause_recall

### 3. Final answer evaluation

Для нового experiment рассчитать:

- answer_point_similarity
- answer_point_recall
- answer_type_match

### 4. Operational evaluation

Для нового experiment рассчитать:

- fallback_triggered
- ungrounded_flag
- technical_error
- empty_answer
- empty_citations

### 5. Retrieval-only run

Запустить отдельный retrieval experiment
на том же dataset:

`python scripts/run_langsmith_retrieval_evaluation.py`

Метрики:

- retrieval_doc_recall
- retrieval_clause_recall

### 6. LangSmith Compare

Сравнить:

новый full-pipeline experiment
vs
`cbr-llp-copilot-baseline-d0d10cd3`

и отдельно:

новый retrieval experiment
vs
`cbr-llp-copilot-retrieval-69a00985`

---

## MVP acceptance criteria

| Metric | Acceptance threshold |
|---|---:|
| retrieval_doc_recall | >= 0.90 |
| retrieval_clause_recall | >= 0.90 |
| citation_doc_recall | >= 0.90 |
| citation_clause_recall | >= 0.60 |
| answer_point_similarity | >= 0.80 |
| answer_point_recall | >= 0.90 |
| answer_type_match | >= 0.90 |
| fallback_triggered | <= 0.10 |
| ungrounded_flag | <= 0.10 |
| technical_error | 0.00 |
| empty_answer | 0.00 |
| empty_citations | 0.00 |

---

## Regression rule

Изменение считается не вызвавшим критической регрессии,
если:

1. все обязательные MVP acceptance criteria выполняются;
2. ни одна ключевая метрика не падает ниже установленного порога;
3. technical_error, empty_answer и empty_citations остаются 0.00.

Если метрика ухудшилась относительно baseline,
но остается выше acceptance threshold,
это фиксируется как degradation, но не как blocking regression.

Если метрика опустилась ниже acceptance threshold,
изменение считается regression и требует анализа.

---

## Key baseline values

| Metric | Baseline |
|---|---:|
| retrieval_doc_recall | 1.00 |
| retrieval_clause_recall | 1.00 |
| citation_doc_recall | 1.00 |
| citation_clause_recall | 0.60 |
| answer_point_similarity | 0.87 |
| answer_point_recall | 1.00 |
| answer_type_match | 1.00 |
| fallback_triggered | 0.00 |
| ungrounded_flag | 0.00 |
| technical_error | 0.00 |
| empty_answer | 0.00 |
| empty_citations | 0.00 |

---

## Interpretation

Regression check отвечает на два вопроса:

1. Проходит ли новая версия минимальные критерии качества MVP?
2. Улучшилась или ухудшилась она относительно зафиксированного baseline?

Acceptance threshold и baseline — разные понятия.

Baseline показывает результат контрольной версии.

Acceptance threshold определяет минимально допустимый уровень качества.