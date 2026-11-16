
"""
Search and answer mode configuration for CBR Loan Loss Provisions Copilot.

This module contains UI-facing labels and internal keys for:
- source filtering modes;
- answer generation modes.

At this stage the config only defines the contract.
Actual filtering logic will be connected in the next steps of Stage 9.
"""


# ---------------------------------------------------------------------------
# Source modes
# ---------------------------------------------------------------------------

SOURCE_MODE_ALL = "all"
SOURCE_MODE_590P = "590p"
SOURCE_MODE_611P = "611p"
SOURCE_MODE_590P_EXPLANATIONS = "590p_explanations"
SOURCE_MODE_611P_EXPLANATIONS = "611p_explanations"
SOURCE_MODE_PRIMARY = "primary"
SOURCE_MODE_SECONDARY = "secondary"


SOURCE_MODES = {
    SOURCE_MODE_ALL: {
        "label": "Весь корпус",
        "description": "Поиск по 590-П, 611-П и разъяснениям.",
        "sources": ["590P", "611P", "590P_EXPL", "611P_EXPL"],
    },
    SOURCE_MODE_590P: {
        "label": "Только 590-П",
        "description": "Поиск только по тексту Положения 590-П.",
        "sources": ["590P"],
    },
    SOURCE_MODE_611P: {
        "label": "Только 611-П",
        "description": "Поиск только по тексту Положения 611-П.",
        "sources": ["611P"],
    },
    SOURCE_MODE_590P_EXPLANATIONS: {
        "label": "Только разъяснения 590-П",
        "description": "Поиск только по разъяснениям Банка России к 590-П.",
        "sources": ["590P_EXPL"],
    },
    SOURCE_MODE_611P_EXPLANATIONS: {
        "label": "Только разъяснения 611-П",
        "description": "Поиск только по разъяснениям Банка России к 611-П.",
        "sources": ["611P_EXPL"],
    },
    SOURCE_MODE_PRIMARY: {
        "label": "Только нормативные документы",
        "description": "Поиск только по первичным нормативным источникам: 590-П и 611-П.",
        "sources": ["590P", "611P"],
    },
    SOURCE_MODE_SECONDARY: {
        "label": "Только разъяснения",
        "description": "Поиск только по разъяснениям к 590-П и 611-П.",
        "sources": ["590P_EXPL", "611P_EXPL"],
    },
}


DEFAULT_SOURCE_MODE = SOURCE_MODE_ALL


# ---------------------------------------------------------------------------
# Answer modes
# ---------------------------------------------------------------------------

ANSWER_MODE_NORM_ONLY = "norm_only"
ANSWER_MODE_NORM_PLUS_EXPLANATION = "norm_plus_explanation"
ANSWER_MODE_FULL_CORPUS = "full_corpus"


ANSWER_MODES = {
    ANSWER_MODE_NORM_ONLY: {
        "label": "Только норма",
        "description": "Ответ строится преимущественно по первичным нормативным источникам.",
    },
    ANSWER_MODE_NORM_PLUS_EXPLANATION: {
        "label": "Норма + разъяснение",
        "description": "Ответ строится по логике: сначала норма, затем разъяснение.",
    },
    ANSWER_MODE_FULL_CORPUS: {
        "label": "Поиск по всему корпусу",
        "description": "Широкий поиск по доступным источникам корпуса.",
    },
}


DEFAULT_ANSWER_MODE = ANSWER_MODE_NORM_PLUS_EXPLANATION


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_source_mode_labels() -> list[str]:
    """
    Return source mode labels for UI selectbox.
    """
    return [mode_config["label"] for mode_config in SOURCE_MODES.values()]


def get_answer_mode_labels() -> list[str]:
    """
    Return answer mode labels for UI selectbox.
    """
    return [mode_config["label"] for mode_config in ANSWER_MODES.values()]


def get_source_mode_key_by_label(label: str) -> str:
    """
    Convert UI label to internal source mode key.
    """
    for mode_key, mode_config in SOURCE_MODES.items():
        if mode_config["label"] == label:
            return mode_key

    return DEFAULT_SOURCE_MODE


def get_answer_mode_key_by_label(label: str) -> str:
    """
    Convert UI label to internal answer mode key.
    """
    for mode_key, mode_config in ANSWER_MODES.items():
        if mode_config["label"] == label:
            return mode_key

    return DEFAULT_ANSWER_MODE


def get_sources_for_mode(source_mode: str) -> list[str]:
    """
    Return list of internal source identifiers for selected source mode.
    """
    return SOURCE_MODES.get(
        source_mode,
        SOURCE_MODES[DEFAULT_SOURCE_MODE],
    )["sources"]

def normalize_source_mode(source_mode: str) -> str:
    """
    Validate source_mode and return a safe internal source mode key.
    """
    if source_mode in SOURCE_MODES:
        return source_mode

    return DEFAULT_SOURCE_MODE


def normalize_answer_mode(answer_mode: str) -> str:
    """
    Validate answer_mode and return a safe internal answer mode key.
    """
    if answer_mode in ANSWER_MODES:
        return answer_mode

    return DEFAULT_ANSWER_MODE