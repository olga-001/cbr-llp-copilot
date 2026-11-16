from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import urllib.request
import zipfile
from pathlib import Path

import chromadb

# ---------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_FILENAME = "Vikhr-Llama-3.2-1B-Q4_K_M.gguf"
MODEL_PATH = PROJECT_ROOT / "data" / "models" / MODEL_FILENAME

MODEL_REVISION = "dd9d8ec10bf0eb62019b4347ad7f48687b14ab75"

MODEL_URL = (
    "https://huggingface.co/"
    "Vikhrmodels/Vikhr-Llama-3.2-1B-instruct-GGUF/"
    f"resolve/{MODEL_REVISION}/Vikhr-Llama-3.2-1B-Q4_K_M.gguf"
)

MODEL_SHA256 = (
    "a85c0ef4b6476ff5fcf2c961226234a"
    "b7008d17087c2152416e61a9b7e74b32d"
)

CHROMA_DIR = PROJECT_ROOT / "data" / "chroma"
CHROMA_ARCHIVE = (
    PROJECT_ROOT / "deployment" / "chroma_index_mvp.zip"
)

COLLECTION_NAME = "cbr_llp_copilot"
EXPECTED_DOCUMENT_COUNT = 583

_BOOTSTRAP_LOCK = threading.Lock()


# ---------------------------------------------------------
# File integrity
# ---------------------------------------------------------

def calculate_sha256(file_path: Path) -> str:
    digest = hashlib.sha256()

    with file_path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


# ---------------------------------------------------------
# GGUF model preparation
# ---------------------------------------------------------

def ensure_model() -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    if MODEL_PATH.is_file():
        if calculate_sha256(MODEL_PATH) != MODEL_SHA256:
            raise RuntimeError(
                "GGUF-модель существует, но её SHA-256 "
                "не совпадает с ожидаемым. "
                f"Проверьте файл: {MODEL_PATH}"
            )

        return

    # Download into a temporary file first.
    # The final model path is used only after validation.
    temporary_path = MODEL_PATH.with_name(
        MODEL_PATH.name + ".download"
    )

    try:
        request = urllib.request.Request(
            MODEL_URL,
            headers={"User-Agent": "CBR-LLP-Copilot/1.0"},
        )

        with urllib.request.urlopen(
            request, timeout=120
        ) as response:
            with temporary_path.open("wb") as output:
                shutil.copyfileobj(response, output)

        downloaded_hash = calculate_sha256(temporary_path)

        if downloaded_hash != MODEL_SHA256:
            raise RuntimeError(
                "SHA-256 скачанной GGUF-модели не совпадает "
                "с ожидаемым значением."
            )

        os.replace(temporary_path, MODEL_PATH)

    finally:
        temporary_path.unlink(missing_ok=True)


# ---------------------------------------------------------
# Chroma integrity check
# ---------------------------------------------------------

def validate_chroma(directory: Path) -> None:
    if not (directory / "chroma.sqlite3").is_file():
        raise RuntimeError(
            f"Не найден файл chroma.sqlite3: {directory}"
        )

    client = chromadb.PersistentClient(path=str(directory))

    try:
        collection = client.get_collection(
            name=COLLECTION_NAME
        )
    except Exception as exc:
        raise RuntimeError(
            f"Коллекция {COLLECTION_NAME} не найдена."
        ) from exc

    actual_count = collection.count()

    if actual_count != EXPECTED_DOCUMENT_COUNT:
        raise RuntimeError(
            "Некорректное количество записей Chroma: "
            f"ожидалось {EXPECTED_DOCUMENT_COUNT}, "
            f"получено {actual_count}."
        )


def validate_chroma_in_subprocess(directory: Path) -> None:
    """
    Validate a restored Chroma index in a separate process.

    This prevents SQLite file locks from blocking
    directory relocation on Windows.
    """

    validation_code = """
import sys
import chromadb

directory = sys.argv[1]
collection_name = sys.argv[2]
expected_count = int(sys.argv[3])

client = chromadb.PersistentClient(path=directory)
collection = client.get_collection(name=collection_name)

actual_count = collection.count()

if actual_count != expected_count:
    raise RuntimeError(
        f"Chroma count mismatch: "
        f"expected {expected_count}, got {actual_count}"
    )
"""

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            validation_code,
            str(directory),
            COLLECTION_NAME,
            str(EXPECTED_DOCUMENT_COUNT),
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Проверка восстановленного Chroma-индекса "
            "завершилась ошибкой:\n"
            + result.stderr
        )

# ---------------------------------------------------------
# Chroma archive restoration
# ---------------------------------------------------------

def ensure_chroma() -> None:
    # Existing index must be valid.
    # Never overwrite it automatically.
    if CHROMA_DIR.exists():
        validate_chroma(CHROMA_DIR)
        return

    if not CHROMA_ARCHIVE.is_file():
        raise RuntimeError(
            f"Архив Chroma не найден: {CHROMA_ARCHIVE}"
        )

    CHROMA_DIR.parent.mkdir(
        parents=True, exist_ok=True
    )

    # Extract and validate in a separate temporary directory.
    with tempfile.TemporaryDirectory(
        prefix="chroma_bootstrap_",
        dir=str(CHROMA_DIR.parent),
    ) as temporary_directory:

        temporary_root = Path(temporary_directory)
        restored_dir = temporary_root / "chroma"
        restored_dir.mkdir()

        with zipfile.ZipFile(CHROMA_ARCHIVE, "r") as archive:
            for member in archive.infolist():
                # Our archive must contain files under chroma/.
                relative_path = Path(
                    member.filename.replace("\\", "/")
                )

                if (
                    not relative_path.parts
                    or relative_path.parts[0] != "chroma"
                    or ".." in relative_path.parts
                ):
                    raise RuntimeError(
                        "Некорректная структура Chroma ZIP."
                    )

                if member.is_dir():
                    continue

                destination = (
                    restored_dir.joinpath(
                        *relative_path.parts[1:]
                    )
                )

                destination.parent.mkdir(
                    parents=True, exist_ok=True
                )

                with archive.open(member) as source:
                    with destination.open("wb") as output:
                        shutil.copyfileobj(source, output)

        validate_chroma_in_subprocess(restored_dir)

        # Move a fully validated index into its final location.
        if CHROMA_DIR.exists():
            validate_chroma(CHROMA_DIR)
            return

        restored_dir.rename(CHROMA_DIR)


# ---------------------------------------------------------
# Public entry point
# ---------------------------------------------------------

def prepare_runtime() -> None:
    """
    Prepare local artifacts required by the RAG application.

    - Validate or download the GGUF model.
    - Validate or restore the persistent Chroma index.
    - Never rebuild the index from source documents.
    """
    with _BOOTSTRAP_LOCK:
        ensure_chroma()
        ensure_model()


if __name__ == "__main__":
    prepare_runtime()
    print("Deployment artifacts are ready.")
