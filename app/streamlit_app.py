
import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv


# ---------------------------------------------------------
# Project root
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------

load_dotenv(
    dotenv_path=PROJECT_ROOT / ".env",
    override=False,
)

# ---------------------------------------------------------
# Project imports
# ---------------------------------------------------------

from rag.runtime_diagnostics import measure_stage, log_event
from deployment.bootstrap import prepare_runtime
from rag.answer_pipeline import answer_question

from rag.search_config import (
    DEFAULT_ANSWER_MODE,
    DEFAULT_SOURCE_MODE,
    ANSWER_MODES,
    SOURCE_MODES,
    get_answer_mode_key_by_label,
    get_answer_mode_labels,
    get_source_mode_key_by_label,
    get_source_mode_labels,
)

from rag.export_utils import (
    prepare_export_data,
    format_copy_ready,
)

from rag.export_docx import create_docx


# ---------------------------------------------------------
# Streamlit page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="CBR Loan Loss Provisions Copilot",
    page_icon="📘",
    layout="wide",
)

# ---------------------------------------------------------
# Deployment bootstrap
# ---------------------------------------------------------


@st.cache_resource(show_spinner=False)
def initialize_runtime():
    log_event(
        f"RUNTIME_CONFIG | "
        f"os_cpu_count={os.cpu_count()} | "
        f"OMP_NUM_THREADS={os.getenv('OMP_NUM_THREADS', 'N/A')} | "
        f"MKL_NUM_THREADS={os.getenv('MKL_NUM_THREADS', 'N/A')} | "
        f"LLAMA_N_CTX={os.getenv('LLAMA_N_CTX', 'default')} | "
        f"LLAMA_N_THREADS={os.getenv('LLAMA_N_THREADS', 'default')} | "
        f"LLAMA_N_BATCH={os.getenv('LLAMA_N_BATCH', 'default')} | "
        f"LLAMA_MAX_TOKENS={os.getenv('LLAMA_MAX_TOKENS', 'default')} | "
        f"LLAMA_N_GPU_LAYERS={os.getenv('LLAMA_N_GPU_LAYERS', 'default')}"
    )

    with measure_stage("app.runtime_preparation"):
        prepare_runtime()

    return True


try:
    with st.spinner("Подготовка файлов приложения..."):
        initialize_runtime()
except Exception as exc:
    st.error(
        "Не удалось подготовить файлы приложения "
        "(GGUF-модель или Chroma-индекс)."
    )
    st.exception(exc)
    st.stop()

# ---------------------------------------------------------
# Session state
# ---------------------------------------------------------

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "last_question" not in st.session_state:
    st.session_state.last_question = None

if "export_text" not in st.session_state:
    st.session_state.export_text = None

if "export_docx" not in st.session_state:
    st.session_state.export_docx = None


# ---------------------------------------------------------
# Application header
# ---------------------------------------------------------

st.title("📘 CBR Loan Loss Provisions Copilot")

st.markdown(
    """
    Инструмент для поиска и формирования ответов по нормативным
    документам Банка России о формировании резервов
    на возможные потери по ссудам.
    """
)

st.warning(
    "Демонстрационная версия. Ответы формируются на основе загруженного корпуса "
    "документов и не предназначены для профессионального "
    "использования в работе."
)


# ---------------------------------------------------------
# User question
# ---------------------------------------------------------

question = st.text_area(
    "Введите вопрос по нормативке:",
    placeholder=(
        "Например: Какая категория качества, если финансовое "
        "положение среднее и обслуживание долга среднее?"
    ),
    height=120,
)


# ---------------------------------------------------------
# Search / answer settings
# ---------------------------------------------------------

st.subheader("Параметры поиска и ответа")

source_mode_labels = get_source_mode_labels()
answer_mode_labels = get_answer_mode_labels()

default_source_label = SOURCE_MODES[
    DEFAULT_SOURCE_MODE
]["label"]

default_answer_label = ANSWER_MODES[
    DEFAULT_ANSWER_MODE
]["label"]

col_source, col_answer = st.columns(2)

with col_source:
    selected_source_mode_label = st.selectbox(
        "Источник поиска",
        options=source_mode_labels,
        index=source_mode_labels.index(default_source_label),
    )

with col_answer:
    selected_answer_mode_label = st.selectbox(
        "Режим ответа",
        options=answer_mode_labels,
        index=answer_mode_labels.index(default_answer_label),
    )

selected_source_mode = get_source_mode_key_by_label(
    selected_source_mode_label
)

selected_answer_mode = get_answer_mode_key_by_label(
    selected_answer_mode_label
)

with st.expander("Описание выбранных параметров"):

    st.write(
        "**Источник поиска:**",
        selected_source_mode_label,
    )

    st.write(
        "**Режим ответа:**",
        selected_answer_mode_label,
    )


# ---------------------------------------------------------
# Run Answer Pipeline
# ---------------------------------------------------------

run_button = st.button(
    "Получить ответ",
    type="primary",
)

if run_button:

    if not question.strip():
        st.error("Введите вопрос.")

    else:

        # Clear previous result before a new request
        st.session_state.last_result = None
        st.session_state.last_question = None
        st.session_state.export_text = None
        st.session_state.export_docx = None

        with st.spinner(
            "Ищу релевантные нормы и формирую ответ..."
        ):

            try:


                with measure_stage("app.request_total"):

                    # 1. Generate answer
                    with measure_stage("app.answer_question"):
                        result = answer_question(
                            question=question,
                            source_mode=selected_source_mode,
                            answer_mode=selected_answer_mode,
                        )

                    # 2. Save answer and original question
                    st.session_state.last_result = result
                    st.session_state.last_question = question

                    # 3. Prepare export data
                    with measure_stage("app.export_data"):
                        export_data = prepare_export_data(
                            question=question,
                            result=result,
                        )

                    # 4. Prepare copy-ready text
                    with measure_stage("app.export_text"):
                        st.session_state.export_text = (
                            format_copy_ready(export_data)
                        )

                    # 5. Prepare DOCX in memory
                    with measure_stage("app.export_docx"):
                        docx_buffer = create_docx(export_data)

                        st.session_state.export_docx = (
                            docx_buffer.getvalue()
                        )

            except Exception as e:

                st.error(
                    "Произошла ошибка при формировании ответа."
                )

                st.exception(e)


# ---------------------------------------------------------
# Display saved answer
# ---------------------------------------------------------

result = st.session_state.last_result

if result is not None:

    st.divider()

    st.subheader("Ответ")

    answer = getattr(result, "answer", None)

    if answer:
        st.write(answer)
    else:
        st.warning("Ответ не был сформирован.")

    # -----------------------------------------------------
    # Answer status
    # -----------------------------------------------------

    st.divider()

    st.subheader("Статус ответа")

    grounded = getattr(result, "grounded", None)
    answer_mode = getattr(result, "answer_mode", None)
    notes = getattr(result, "notes", None)

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Grounded:**", grounded)

    with col2:
        st.write("**Answer mode:**", answer_mode)

    if notes:
        st.info(notes)

    # -----------------------------------------------------
    # Citations
    # -----------------------------------------------------

    st.divider()

    st.subheader("Источники / citations")

    citations = getattr(result, "citations", None)

    if citations:

        for i, citation in enumerate(
            citations,
            start=1,
        ):

            with st.expander(f"Источник {i}"):
                st.write(citation)

    else:
        st.warning("Citations не найдены.")

    # -----------------------------------------------------
    # Export section
    # -----------------------------------------------------

    st.divider()

    st.subheader("Экспорт справки")

    st.caption(
        "Справка формируется на основе последнего "
        "полученного ответа без повторного обращения к LLM."
    )

    # DOCX download
    if st.session_state.export_docx is not None:

        st.download_button(
            label="📄 Скачать DOCX",
            data=st.session_state.export_docx,
            file_name="CBR_LLP_Copilot_Answer.docx",
            mime=(
                "application/vnd.openxmlformats-officedocument."
                "wordprocessingml.document"
            ),
            on_click="ignore",
            type="primary",
        )

    # Copy-ready text
    if st.session_state.export_text is not None:

        with st.expander("📋 Копировать справку"):

            st.caption(
                "Нажмите значок копирования в правом "
                "верхнем углу блока ниже."
            )

            st.code(
                st.session_state.export_text,
                language=None,
            )
