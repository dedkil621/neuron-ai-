# -*- coding: utf-8 -*-
"""НЕЙРОН — Telegram AI assistant. Python 3.10+."""

import os
import time
import sqlite3
import logging
import requests

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
OWNER_ID_RAW = os.getenv("OWNER_ID", "").strip()
OWNER_ID = int(OWNER_ID_RAW) if OWNER_ID_RAW.isdigit() else None

MODEL = "openai/gpt-oss-120b"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DB_PATH = os.getenv("DB_PATH", "neuron.db")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger("neuron")
http = requests.Session()


def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS history (
                user_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS knowledge (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                created_by INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)


def tg(method, data=None, timeout=40):
    response = http.post(
        f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/{method}",
        json=data or {},
        timeout=timeout
    )
    response.raise_for_status()
    result = response.json()
    if not result.get("ok"):
        raise RuntimeError(result.get("description", "Telegram API error"))
    return result.get("result")


def send(chat_id, text, reply_markup=None):
    payload = {"chat_id": chat_id, "text": str(text)[:4000]}
    if reply_markup:
        payload["reply_markup"] = reply_markup
    return tg("sendMessage", payload)


def is_owner(user_id):
    return OWNER_ID is not None and int(user_id) == OWNER_ID


def remember(user_id, role, content):
    with db() as conn:
        conn.execute(
            "INSERT INTO history(user_id, role, content) VALUES (?, ?, ?)",
            (int(user_id), role, content)
        )
        # Keep the most recent 20 messages for each user.
        conn.execute("""
            DELETE FROM history
            WHERE user_id = ? AND id NOT IN (
                SELECT id FROM history WHERE user_id = ?
                ORDER BY id DESC LIMIT 20
            )
        """, (int(user_id), int(user_id)))


def get_history(user_id):
    with db() as conn:
        rows = conn.execute("""
            SELECT role, content FROM history
            WHERE user_id = ? ORDER BY id DESC LIMIT 20
        """, (int(user_id),)).fetchall()
    return [{"role": row["role"], "content": row["content"]} for row in reversed(rows)]


def get_knowledge(limit=25):
    with db() as conn:
        rows = conn.execute("""
            SELECT question, answer FROM knowledge
            ORDER BY id DESC LIMIT ?
        """, (limit,)).fetchall()
    return list(reversed(rows))


def ask_ai(user_id, user_text):
    history = get_history(user_id)
    knowledge = get_knowledge()
    knowledge_text = ""
    if knowledge:
        knowledge_text = "\n\nОбщие знания, добавленные создателем:\n" + "\n".join(
            f"Вопрос: {row['question']}\nОтвет: {row['answer']}"
            for row in knowledge
        )

    system_prompt = (
        "Ты НЕЙРОН — умный, живой и дружелюбный Telegram-помощник. "
        "Общайся естественно, по-русски, если пользователь пишет по-русски. "
        "Не повторяй один и тот же шаблонный ответ. Учитывай контекст беседы. "
        "Объясняй понятно и по делу. Если не знаешь ответа — честно скажи. "
        "Не выдумывай факты и не утверждай, что умеешь выполнять действия, "
        "которых на самом деле не выполняешь."
        + knowledge_text
    )

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_text})

    response = http.post(
        GROQ_URL,
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": MODEL,
            "messages": messages,
            "temperature": 0.8,
            "max_tokens": 1400
        },
        timeout=90
    )
    if response.status_code != 200:
        try:
            details = response.json().get("error", {}).get("message", response.text)
        except Exception:
            details = response.text
        raise RuntimeError(f"Groq API {response.status_code}: {details}")

    data = response.json()
    answer = data["choices"][0]["message"]["content"].strip()
    if not answer:
        answer = "Не получилось сформировать ответ. Попробуй написать иначе."
    remember(user_id, "user", user_text)
    remember(user_id, "assistant", answer)
    return answer


def handle_message(message):
    chat = message.get("chat", {})
    sender = message.get("from", {})
    chat_id = chat.get("id")
    user_id = sender.get("id")
    text = (message.get("text") or "").strip()

    if not chat_id or not user_id or not text:
        return

    if text.startswith("/start"):
        send(chat_id,
             "Привет! Я НЕЙРОН — ИИ-помощник.\n\n"
             "Просто напиши мне сообщение, и я отвечу.\n"
             "Команды: /help и /id")
        return

    if text.startswith("/help"):
        send(chat_id,
             "Я умею отвечать на вопросы и поддерживать диалог.\n\n"
             "/start — начать\n"
             "/id — показать твой Telegram ID\n"
             "/reset — очистить память диалога\n"
             "/help — помощь")
        return

    if text.startswith("/id"):
        send(chat_id, f"Твой Telegram ID: {user_id}")
        return

    if text.startswith("/reset"):
        with db() as conn:
            conn.execute("DELETE FROM history WHERE user_id = ?", (int(user_id),))
        send(chat_id, "История нашего диалога очищена.")
        return

    if text.startswith("/teach"):
        if not is_owner(user_id):
            send(chat_id, "Эта команда доступна только создателю.")
            return
        payload = text[len("/teach"):].strip()
        if "|" not in payload:
            send(chat_id, "Формат: /teach вопрос | правильный ответ")
            return
        question, answer = [part.strip() for part in payload.split("|", 1)]
        if not question or not answer:
            send(chat_id, "Заполни и вопрос, и ответ.")
            return
        with db() as conn:
            conn.execute(
                "INSERT INTO knowledge(question, answer, created_by) VALUES (?, ?, ?)",
                (question, answer, int(user_id))
            )
        send(chat_id, "Принято. Добавил это в общую базу знаний НЕЙРОНа.")
        return

    if text.startswith("/knowledge"):
        if not is_owner(user_id):
            send(chat_id, "Эта команда доступна только создателю.")
            return
        rows = get_knowledge(30)
        if not rows:
            send(chat_id, "База знаний пока пустая.")
            return
        output = ["Последние знания НЕЙРОНа:"]
        for i, row in enumerate(rows, 1):
            output.append(f"{i}. {row['question']} → {row['answer']}")
        send(chat_id, "\n".join(output))
        return

    if text.startswith("/clear_knowledge"):
        if not is_owner(user_id):
            send(chat_id, "Эта команда доступна только создателю.")
            return
        with db() as conn:
            conn.execute("DELETE FROM knowledge")
        send(chat_id, "Общая база знаний очищена.")
        return

    try:
        tg("sendChatAction", {"chat_id": chat_id, "action": "typing"})
        answer = ask_ai(user_id, text)
        send(chat_id, answer)
    except requests.Timeout:
        send(chat_id, "ИИ отвечает слишком долго. Попробуй ещё раз.")
    except Exception as exc:
        log.exception("Message processing failed")
        send(chat_id, f"Ошибка ИИ: {str(exc)[:700]}")


def main():
    if not TELEGRAM_TOKEN:
        raise SystemExit("Не задана переменная TELEGRAM_TOKEN")
    if not GROQ_API_KEY:
        raise SystemExit("Не задана переменная GROQ_API_KEY")
    init_db()
    log.info("НЕЙРОН запущен. Модель: %s", MODEL)

    me = tg("getMe")
    log.info("Бот: @%s", me.get("username", "unknown"))
    offset = 0

    while True:
        try:
            updates = tg("getUpdates", {
                "offset": offset,
                "timeout": 25,
                "allowed_updates": ["message"]
            }, timeout=35)
            for update in updates:
                offset = update["update_id"] + 1
                if "message" in update:
                    handle_message(update["message"])
        except KeyboardInterrupt:
            log.info("Остановка по запросу.")
            break
        except Exception:
            log.exception("Ошибка polling; повтор через 5 секунд")
            time.sleep(5)


if __name__ == "__main__":
    main()
