# CBR Loan Loss Provisions Copilot
## Stage 10 — Baseline Evaluation

### Baseline status

Baseline зафиксирован по результатам контрольного evaluation-прогона
на 10 кейсах dataset:

`cbr-llp-copilot-mvp-eval`

Основной LangSmith experiment:

`cbr-llp-copilot-baseline-d0d10cd3`

Retrieval experiment:

`cbr-llp-copilot-retrieval-69a00985`

---

## Baseline configuration

### Pipeline

- source_mode: `all`
- answer_mode: `norm_plus_explanation`
- primary_k: `3`
- secondary_k: `2`
- tertiary_k: `0`
- use_combined_fallback: `True`
- max_citations: `5`

### Retrieval

Основной режим поиска:

`source_mode = all`

Доступные источники:

- `590P`
- `611P`
- `590P_EXPL`
- `611P_EXPL`

### Embeddings

Model:

`intfloat/multilingual-e5-base`

### Local LLM

Model:

`Vikhr-Llama-3.2-1B-instruct-GGUF`

Quantization:

`Q4_K_M`

Inference:

- local
- CPU
- llama.cpp

---

## Evaluation dataset

Количество контрольных кейсов:

`10`

Каждый кейс содержит:

- case_id
- question
- expected_doc_ids
- expected_clauses
- expected_answer_points
- expected_answer_type

---

## Baseline metrics

| Block | Metric | Baseline |
|---|---|---:|
| Retrieval | retrieval_doc_recall | 1.00 |
| Retrieval | retrieval_clause_recall | 1.00 |
| Citations | citation_doc_recall | 1.00 |
| Citations | citation_clause_recall | 0.60 |
| Final answer | answer_point_similarity | 0.87 |
| Final answer | answer_point_recall | 1.00 |
| Final answer | answer_type_match | 1.00 |
| Operational | fallback_triggered | 0.00 |
| Operational | ungrounded_flag | 0.00 |
| Operational | technical_error | 0.00 |
| Operational | empty_answer | 0.00 |
| Operational | empty_citations | 0.00 |

---

## Retrieval mode comparison

| source_mode | retrieval_doc_recall | retrieval_clause_recall | Latency P50 |
|---|---:|---:|---:|
| all | 1.00 | 1.00 | ~0.266 s |
| primary | 0.85 | 0.60 | ~0.261 s |
| secondary | 0.15 | 0.60 | ~0.155 s |

---

## Baseline interpretation

### Strengths

1. Retrieval на контрольном наборе находит все ожидаемые документы.
2. Retrieval находит все ожидаемые нормативные пункты.
3. Все итоговые ответы покрывают ожидаемые смысловые пункты
   при similarity threshold = 0.80.
4. Средняя semantic similarity итоговых ответов к reference points = 0.87.
5. Все ответы соответствуют ожидаемому типу.
6. Fallback и технические ошибки на контрольном наборе отсутствуют.
7. Пустых ответов и пустых citations нет.

### Main weakness

Основное выявленное слабое место baseline:

`citation_clause_recall = 0.60`

Retrieval находит необходимые нормативные пункты,
но citation layer не всегда отражает ожидаемый пункт
в финальных citations.

Проблемные кейсы:

- EVAL_003
- EVAL_004
- EVAL_005
- EVAL_007

### Search mode conclusion

Для MVP основным режимом остается:

- `source_mode = all`
- `answer_mode = norm_plus_explanation`

Combined retrieval по нормативным документам и разъяснениям
показывает лучший результат на текущем evaluation dataset.

---

## Baseline usage

Этот baseline является контрольной точкой для последующих изменений:

- retrieval logic;
- reranking;
- metadata;
- chunking;
- prompts;
- citation generation;
- LLM;
- corpus composition;
- search parameters.

После существенного изменения системы необходимо повторно
запустить тот же evaluation dataset и сравнить новые метрики
с данным baseline.

Baseline считается улучшенным, если ключевые метрики
не деградируют, а выявленные слабые места улучшаются.