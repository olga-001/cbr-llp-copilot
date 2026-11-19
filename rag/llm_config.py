from pathlib import Path
import os


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_GGUF_MODEL_PATH = (
    PROJECT_ROOT
    / "data"
    / "models"
    / "Vikhr-Llama-3.2-1B-Q4_K_M.gguf"
)

GGUF_MODEL_PATH = Path(
    os.getenv("GGUF_MODEL_PATH", str(DEFAULT_GGUF_MODEL_PATH))
)

# Увеличиваем окно контекста, но не слишком агрессивно для 8 GB RAM
LLAMA_N_CTX = int(os.getenv("LLAMA_N_CTX", "4096"))

# Чуть уменьшаем генерацию, чтобы не съедать лишнее окно
LLAMA_MAX_TOKENS = int(os.getenv("LLAMA_MAX_TOKENS", "350"))

LLAMA_TEMPERATURE = float(os.getenv("LLAMA_TEMPERATURE", "0.3"))
LLAMA_TOP_P = float(os.getenv("LLAMA_TOP_P", "0.95"))
LLAMA_REPEAT_PENALTY = float(os.getenv("LLAMA_REPEAT_PENALTY", "1.05"))

LLAMA_N_GPU_LAYERS = int(os.getenv("LLAMA_N_GPU_LAYERS", "0"))

LLAMA_N_THREADS = int(os.getenv("LLAMA_N_THREADS", "6"))
LLAMA_N_THREADS_BATCH = int(
    os.getenv("LLAMA_N_THREADS_BATCH", str(LLAMA_N_THREADS))
)

# Для слабой машины лучше чуть поменьше
LLAMA_N_BATCH = int(os.getenv("LLAMA_N_BATCH", "64"))

LLAMA_VERBOSE = os.getenv("LLAMA_VERBOSE", "false").lower() == "true"