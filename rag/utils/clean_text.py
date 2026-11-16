import re

_PATTERNS = [
    r"^\s*КонсультантПлюс.*$",
    r"^\s*надежная правовая поддержка.*$",
    r"^\s*Документ предоставленКонсультантПлюс.*$",
    r"^\s*Документ предоставлен\s*КонсультантПлюс.*$",
    r"^\s*www\.consultant\.ru.*$",
    r"^\s*Дата сохранения:.*$",
    r"^\s*Страница\s+\d+\s+из\s+\d+.*$",
    r"^\s*Положение Банка России от .* N\s*590-П.*$",
    r"^\s*\(ред\.\s*от.*\)\s*$",
    r'^\s*".*резерв.*"\s*$',  # строка с названием (в кавычках)
    r"^\s*\(вместе с.*\)\s*$",
    r"^\s*\(Зарегистрировано.*\)\s*$",
    ]
_rx = [re.compile(p, flags=re.IGNORECASE) for p in _PATTERNS]

def clean_consultant_noise(text: str) -> str:
    lines = text.splitlines()
    out = []
    for line in lines:
        s = line.strip()
        if any(r.match(s) for r in _rx):
            continue
        out.append(line)
    cleaned = "\n".join(out)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned