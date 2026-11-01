# -*- coding: utf-8 -*-
"""НЕЙРОН — Telegram AI bot for FPS.ms (long polling)."""

import os
import time
import sqlite3
import logging
import requests
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
OWNER_ID_RAW = os.getenv("OWNER_ID", "").strip()
OWNER_ID = int(OWNER_ID_RAW) if OWNER_ID_RAW.isdigit() else None

MODEL = "openai/gpt-oss-120b"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DB_PATH = os.getenv("DB_PATH", "neuron.db")

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(message)s"
)
log = logging.getLogger("neuron")
http = requests.Session()
http.headers.update({"User-Agent": "NEIRON-Telegram-Bot/1.0"})


def db():
    conn = sqlite3.connect(DB_PATH, timeout=20)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with db() as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL
        )""")
        conn.execute("""CREATE TABLE IF NOT EXISTS knowledge (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            created_by INTEGER
        )""")


def telegram(method, data=None, timeout=40):
    if not TELEGRAM_TOKEN:
        raise RuntimeError("Не задан TELEGRAM_TOKEN в файле .env")
    response = http.post(
        f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/{method}",
        json=data or {},
        timeout=timeout
    )
    response.raise_for_status()
    payload = response.json()
    if not payload.get("ok"):
        raise RuntimeError(payload.get("description", "Ошибка Telegram API"))
    return payload.get("result")


def send(chat_id, message):
    telegram("sendMessage", {
        "chat_id": chat_id,
        "text": str(message)[:4000]
    })


def is_owner(user_id):
    return OWNER_ID is not None and int(user_id) == OWNER_ID


def get_history(user_id):
    with db() as conn:
        rows = conn.execute(
            "SELECT role, content FROM history WHERE user_id=? ORDER BY id DESC LIMIT 20",
            (int(user_id),)
        ).fetchall()
    return [{"role": row["role"], "content": row["content"]} for row in reversed(rows)]


def save_history(user_id, role, content):
    with db() as conn:
        conn.execute(
            "INSERT INTO history(user_id, role, content) VALUES(?,?,?)",
            (int(user_id), role, content)
        )
        conn.execute(
            """DELETE FROM history WHERE user_id=? AND id NOT IN
               (SELECT id FROM history WHERE user_id=? ORDER BY id DESC LIMIT 20)""",
            (int(user_id), int(user_id))
        )


def get_knowledge():
    with db() as conn:
        return conn.execute(
            "SELECT question, answer FROM knowledge ORDER BY id DESC LIMIT 25"
        ).fetchall()


def ask_ai(user_id, prompt):
    if not GROQ_API_KEY:
        raise RuntimeError("Не задан GROQ_API_KEY в файле .env")

    facts = get_knowledge()
    extra = ""
    if facts:
        extra = "\n\nЗнания, добавленные создателем:\n" + "\n".join(
            f"Вопрос: {row['question']}\nОтвет: {row['answer']}" for row in facts
        )

    messages = [{
        "role": "system",
        "content": (
            "Ты НЕЙРОН — дружелюбный, умный русскоязычный ИИ-помощник. "
            "Общайся естественно, отвечай по существу, учитывай контекст диалога. "
            "Если не знаешь ответа, честно скажи об этом." + extra
        )
    }]
    messages.extend(get_history(user_id))
    messages.append({"role": "user", "content": prompt})

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
            detail = response.json().get("error", {}).get("message", response.text)
        except Exception:
            detail = response.text
        raise RuntimeError(f"Groq API {response.status_code}: {detail[:500]}")

    answer = response.json()["choices"][0]["message"]["content"].strip()
    if not answer:
        answer = "Не получилось сформировать ответ. Попробуй ещё раз."

    save_history(user_id, "user", prompt)
    save_history(user_id, "assistant", answer)
    return answer


def handle_message(message):
    chat_id = message.get("chat", {}).get("id")
    user_id = message.get("from", {}).get("id")
    text = (message.get("text") or "").strip()

    if not chat_id or not user_id or not text:
        return

    if text.startswith("/start"):
        send(chat_id, "Привет! Я НЕЙРОН — ИИ-помощник. Просто напиши мне сообщение.\nКоманды: /help, /id, /reset")
    elif text.startswith("/help"):
        send(
            chat_id,
            "/start — начать\n"
            "/help — помощь\n"
            "/id — твой Telegram ID\n"
            "/reset — очистить историю\n"
            "/teach вопрос | ответ — добавить знание (создатель)\n"
            "/knowledge — показать базу знаний (создатель)\n"
            "/clear_knowledge — очистить базу (создатель)"
        )
    elif text.startswith("/id"):
        send(chat_id, f"Твой Telegram ID: {user_id}")
    elif text.startswith("/reset"):
        with db() as conn:
            conn.execute("DELETE FROM history WHERE user_id=?", (int(user_id),))
        send(chat_id, "История диалога очищена.")
    elif text.startswith("/teach"):
        if not is_owner(user_id):
            send(chat_id, "Эта команда доступна только создателю.")
            return
        raw = text[len("/teach"):].strip()
        if "|" not in raw:
            send(chat_id, "Формат: /teach вопрос | ответ")
            return
        question, answer = [part.strip() for part in raw.split("|", 1)]
        if not question or not answer:
            send(chat_id, "Нужны и вопрос, и ответ.")
            return
        with db() as conn:
            conn.execute(
                "INSERT INTO knowledge(question, answer, created_by) VALUES(?,?,?)",
                (question, answer, int(user_id))
            )
        send(chat_id, "Готово! Добавил это в общую базу знаний.")
    elif text.startswith("/knowledge"):
        if not is_owner(user_id):
            send(chat_id, "Эта команда доступна только создателю.")
            return
        rows = get_knowledge()
        if not rows:
            send(chat_id, "База знаний пока пустая.")
        else:
            send(chat_id, "\n".join(
                f"{index}. {row['question']} → {row['answer']}"
                for index, row in enumerate(rows, 1)
            ))
    elif text.startswith("/clear_knowledge"):
        if not is_owner(user_id):
            send(chat_id, "Эта команда доступна только создателю.")
            return
        with db() as conn:
            conn.execute("DELETE FROM knowledge")
        send(chat_id, "Общая база знаний очищена.")
    else:
        try:
            telegram("sendChatAction", {"chat_id": chat_id, "action": "typing"})
            send(chat_id, ask_ai(user_id, text))
        except Exception as error:
            log.exception("Не удалось обработать сообщение")
            send(chat_id, f"Ошибка ИИ: {str(error)[:700]}")


def main():
    if not TELEGRAM_TOKEN:
        raise SystemExit("Не найден TELEGRAM_TOKEN. Создай .env и добавь токен.")
    if not GROQ_API_KEY:
        raise SystemExit("Не найден GROQ_API_KEY. Добавь ключ Groq в .env.")

    init_db()

    # FPS.ms free Telegram bot servers use polling, not webhooks.
    telegram("deleteWebhook", {"drop_pending_updates": False})
    log.info("НЕЙРОН запущен. Ожидаю сообщения...")

    offset = None
    while True:
        try:
            params = {"timeout": 25}
            if offset is not None:
                params["offset"] = offset

            response = http.get(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates",
                params=params,
                timeout=35
            )
            response.raise_for_status()
            payload = response.json()
            if not payload.get("ok"):
                raise RuntimeError(payload.get("description", "Ошибка getUpdates"))

            for update in payload.get("result", []):
                offset = update["update_id"] + 1
                try:
                    handle_message(update.get("message", {}))
                except Exception:
                    log.exception("Ошибка обработки обновления")

        except requests.RequestException:
            log.exception("Проблема соединения. Повтор через 5 секунд.")
            time.sleep(5)
        except Exception:
            log.exception("Ошибка цикла бота. Повтор через 5 секунд.")
            time.sleep(5)


if __name__ == "__main__":
    main()
