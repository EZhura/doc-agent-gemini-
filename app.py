import os
import re
import math
import json
from collections import Counter

from google import genai


# ------------------ CONFIG ------------------
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
TOP_K = int(os.getenv("RAG_TOP_K", "3"))
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "12"))

SYSTEM_DOC = (
    "Ты помощник, отвечающий СТРОГО по документу.\n"
    "Правила:\n"
    "1) Используй только информацию из CONTEXT.\n"
    "2) Если в CONTEXT нет ответа — скажи: 'В документе нет точных данных.'\n"
    "3) Не добавляй фактов из внешних знаний.\n"
    "4) Цитаты делай короткими (1–2 предложения) и указывай [FRAGMENT <id>].\n"
)

SYSTEM_EXTRACT = (
    "Ты извлекаешь структуру из документа строго по CONTEXT.\n"
    "Верни JSON со следующими полями:\n"
    "- summary: краткое резюме (1-3 предложения)\n"
    "- key_points: список ключевых тезисов (5-10)\n"
    "- entities: список сущностей (люди/компании/даты/суммы), где каждая сущность: {type, value, fragment_id}\n"
    "Если данных нет — верни пустые списки, но валидный JSON.\n"
    "Никаких комментариев вне JSON."
)

WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+", re.UNICODE)


# ------------------ RAG (mini) ------------------
def tokenize(text: str) -> list[str]:
    return [t.lower() for t in WORD_RE.findall(text)]

def counter_and_norm(text: str) -> tuple[Counter, float]:
    toks = tokenize(text)
    c = Counter(toks)
    norm = math.sqrt(sum(v * v for v in c.values())) if c else 0.0
    return c, norm

def cosine(counter_q: Counter, norm_q: float, counter_d: Counter, norm_d: float) -> float:
    if norm_q == 0.0 or norm_d == 0.0:
        return 0.0
    dot = 0
    for w, v in counter_q.items():
        if w in counter_d:
            dot += v * counter_d[w]
    return dot / (norm_q * norm_d)

def chunk_text(text: str, max_chars: int = 1200, overlap_chars: int = 150) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    res = []
    buf = ""
    for p in paragraphs:
        if not buf:
            buf = p
        elif len(buf) + 2 + len(p) <= max_chars:
            buf += "\n\n" + p
        else:
            res.append(buf)
            tail = buf[-overlap_chars:] if overlap_chars > 0 else ""
            buf = (tail + "\n\n" + p).strip()
    if buf:
        res.append(buf)
    return res


class MiniRAG:
    def __init__(self) -> None:
        self.doc_name: str | None = None
        self.doc_text: str | None = None
        self.chunks: list[str] = []
        self.counters: list[Counter] = []
        self.norms: list[float] = []

    def load_txt(self, path: str) -> str:
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
        except FileNotFoundError:
            return f"Файл не найден: {path}"
        except UnicodeDecodeError:
            return "Не удалось прочитать файл как UTF-8. Сохрани .txt в UTF-8."
        except Exception as e:
            return f"Ошибка чтения файла: {e}"

        if not text.strip():
            return "Файл пустой."

        self.doc_name = path
        self.doc_text = text
        self.chunks = chunk_text(text)
        self.counters, self.norms = [], []
        for ch in self.chunks:
            c, n = counter_and_norm(ch)
            self.counters.append(c)
            self.norms.append(n)

        return f"Загружено: {path}. Фрагментов: {len(self.chunks)}"

    def retrieve(self, query: str, top_k: int = TOP_K) -> list[tuple[float, int, str]]:
        if not self.chunks:
            return []
        cq, nq = counter_and_norm(query)
        scored = []
        for i in range(len(self.chunks)):
            s = cosine(cq, nq, self.counters[i], self.norms[i])
            if s > 0:
                scored.append((s, i, self.chunks[i]))
        scored.sort(reverse=True, key=lambda x: x[0])
        return scored[:top_k]


# ------------------ LLM ------------------
def require_api_key() -> str:
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("Нет GEMINI_API_KEY. Задай переменную окружения (см. README).")
    return key

def make_client() -> genai.Client:
    return genai.Client(api_key=require_api_key())

def build_context_blocks(retrieved: list[tuple[float, int, str]]) -> str:
    if not retrieved:
        return "(похожих фрагментов не найдено)"
    blocks = []
    for score, idx, ch in retrieved:
        blocks.append(f"[FRAGMENT {idx} | score={score:.3f}]\n{ch}")
    return "\n\n---\n\n".join(blocks)

def call_model(client: genai.Client, system: str, context: str, user_text: str) -> str:
    prompt = (
        f"SYSTEM:\n{system}\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"USER:\n{user_text}\n"
    )
    resp = client.models.generate_content(model=MODEL, contents=prompt)
    return (resp.text or "").strip()


# ------------------ APP (CLI) ------------------
def main():
    client = make_client()
    rag = MiniRAG()
    history: list[tuple[str, str]] = []  # ("user"/"assistant", text)

    print("✅ Doc Agent (Gemini + mini-RAG)")
    print("Команды:")
    print("  /load <file.txt>    — загрузить документ")
    print("  /ask <вопрос>       — вопрос по документу")
    print("  /extract            — извлечь структуру (JSON) по документу")
    print("  /history            — показать историю")
    print("  /clear              — очистить историю")
    print("  exit                — выход\n")

    while True:
        cmd = input(">> ").strip()
        if not cmd:
            continue
        if cmd.lower() in ("exit", "quit"):
            break

        if cmd.startswith("/load "):
            path = cmd.split(" ", 1)[1].strip()
            msg = rag.load_txt(path)
            print(msg, "\n")
            continue

        if cmd == "/clear":
            history.clear()
            print("История очищена.\n")
            continue

        if cmd == "/history":
            for role, text in history[-MAX_HISTORY_TURNS:]:
                print(f"{role}: {text}\n")
            continue

        if cmd == "/extract":
            if not rag.doc_text:
                print("Сначала загрузите документ: /load file.txt\n")
                continue
            # берем контекст не по вопросу, а “обобщенный”: можно взять первые N фрагментов,
            # но лучше — сделать “обзор” по нескольким фрагментам. Для MVP: первые 3.
            context = build_context_blocks([(1.0, i, rag.chunks[i]) for i in range(min(3, len(rag.chunks)))])
            answer = call_model(client, SYSTEM_EXTRACT, context, "Извлеки структуру документа.")
            # простая валидация JSON (если сломается — покажем как есть)
            try:
                parsed = json.loads(answer)
                print(json.dumps(parsed, ensure_ascii=False, indent=2), "\n")
            except Exception:
                print(answer, "\n")
            continue

        if cmd.startswith("/ask "):
            if not rag.doc_text:
                print("Сначала загрузите документ: /load file.txt\n")
                continue
            question = cmd.split(" ", 1)[1].strip()
            retrieved = rag.retrieve(question, top_k=TOP_K)
            context = build_context_blocks(retrieved)

            # Добавим очень простую память: последние пары Q/A как “подсказку” (не обязательно, но полезно)
            if history:
                short_hist = "\n".join([f"{r.upper()}: {t}" for r, t in history[-6:]])
                context = f"{context}\n\nDIALOG_HINT:\n{short_hist}"

            answer = call_model(client, SYSTEM_DOC, context, question)
            print(answer, "\n")

            history.append(("user", question))
            history.append(("assistant", answer))
            history[:] = history[-MAX_HISTORY_TURNS:]
            continue

        print("Неизвестная команда. /load /ask /extract /history /clear exit\n")


if __name__ == "__main__":
    main()
