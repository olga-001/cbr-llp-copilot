from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DOCS_DIR = BASE_DIR / "data" / "raw_docs"
XLSX_PATH = RAW_DOCS_DIR / "590P_explanations.xlsx"

SHEET_NAME = "Разъяснения по 590-П"

def main():
    if not XLSX_PATH.exists():
        raise SystemExit(f"Не найден файл: {XLSX_PATH}")

    # Считываем "как есть", без шапки
    df = pd.read_excel(XLSX_PATH, sheet_name=SHEET_NAME, header=None)

    # Ищем строку, где есть заголовки "Вопрос" и "Ответ"
    header_row_idx = None
    for i in range(min(50, len(df))):
        row_values = df.iloc[i].astype(str).tolist()
        if "Вопрос" in row_values and "Ответ" in row_values:
            header_row_idx = i
            break

    if header_row_idx is None:
        raise SystemExit("Не нашёл строку заголовков с 'Вопрос' и 'Ответ'.")

    print("Строка заголовков найдена:", header_row_idx)

    # Берём эту строку как заголовки
    headers = df.iloc[header_row_idx].tolist()

    # Данные начинаются со следующей строки
    data = df.iloc[header_row_idx + 1:].copy()
    data.columns = headers

    # Удаляем полностью пустые строки
    data = data.dropna(how="all")

    # Удаляем колонки без названия (nan) и полностью пустые колонки
    data = data.loc[:, [c for c in data.columns if pd.notna(c)]]
    data = data.dropna(axis=1, how="all")

    # Удаляем строки, где нет вопроса и ответа
    data = data.dropna(subset=["Вопрос", "Ответ"], how="any")

    print("Колонки после нормализации:", list(data.columns))
    print("Строк данных:", len(data))

    # Оставим только ключевые колонки, если они есть
    needed = ["Тематика", "Глава", "Пункты", "Ключевые слова", "Вопрос", "Ответ"]
    existing = [c for c in needed if c in data.columns]

    if "Вопрос" not in existing or "Ответ" not in existing:
        raise SystemExit(f"Не нашёл колонки 'Вопрос'/'Ответ'. Есть: {existing}")

    clean = data[existing].copy()

    # Покажем 3 примера
    for idx, row in clean.head(3).iterrows():
        print("\n" + "-" * 80)
        print("Тематика:", str(row.get("Тематика", "")).strip())
        print("Глава:", str(row.get("Глава", "")).strip())
        print("Пункты:", str(row.get("Пункты", "")).strip())
        print("Ключевые слова:", str(row.get("Ключевые слова", "")).strip())
        print("Вопрос:", str(row["Вопрос"]).strip()[:400])
        print("Ответ:", str(row["Ответ"]).strip()[:800])

if __name__ == "__main__":
    main()