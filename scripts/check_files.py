from pathlib import Path

# Получаем путь к корню проекта
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DOCS_DIR = BASE_DIR / "data" / "raw_docs"

print("Ищу файлы в:", RAW_DOCS_DIR)

if not RAW_DOCS_DIR.exists():
    raise SystemExit("Папка data/raw_docs не найдена. Создай её в корне проекта.")

files = sorted([p.name for p in RAW_DOCS_DIR.iterdir() if p.is_file()])
print("Найдено файлов:", len(files))
for f in files:
    print(" -", f)