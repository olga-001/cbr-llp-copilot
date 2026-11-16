from pathlib import Path

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

try:
    from llama_cpp import Llama
except ImportError as e:
    raise ImportError(
        "Не установлен llama-cpp-python. "
        "Установи его перед запуском этого теста: pip install llama-cpp-python"
    ) from e


def main():
    print("GGUF_MODEL_PATH:", GGUF_MODEL_PATH)
    print("Model exists:", Path(GGUF_MODEL_PATH).exists())

    if not Path(GGUF_MODEL_PATH).exists():
        raise FileNotFoundError(
            f"GGUF-модель не найдена по пути: {GGUF_MODEL_PATH}"
        )

    print("Loading model...")
    llm = Llama(
        model_path=str(GGUF_MODEL_PATH),
        n_ctx=LLAMA_N_CTX,
        n_threads=LLAMA_N_THREADS,
        n_gpu_layers=LLAMA_N_GPU_LAYERS,
        n_batch=LLAMA_N_BATCH,
        verbose=LLAMA_VERBOSE,
    )
    print("Model loaded successfully.")

    prompt = (
        "Ты полезный ассистент. "
        "Ответь кратко по-русски одним предложением: "
        "что такое кредитный риск?"
    )

    print("Generating...")
    result = llm(
        prompt,
        max_tokens=LLAMA_MAX_TOKENS,
        temperature=LLAMA_TEMPERATURE,
        top_p=LLAMA_TOP_P,
        repeat_penalty=LLAMA_REPEAT_PENALTY,
        stop=["</s>", "<|eot_id|>"],
    )

    text = result["choices"][0]["text"].strip()

    print("=" * 80)
    print("MODEL RESPONSE:")
    print(text)
    print("=" * 80)


if __name__ == "__main__":
    main()