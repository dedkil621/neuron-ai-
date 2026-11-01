# -*- coding: utf-8 -*-
"""НЕЙРОН — Telegram AI bot for Render Web Service (webhook mode)."""

import os
import sqlite3
import logging
import requests
from flask import Flask, request, abort

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
OWNER_ID_RAW = os.getenv("OWNER_ID", "").strip()
OWNER_ID = int(OWNER_ID_RAW) if OWNER_ID_RAW.isdigit() else None
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "").strip()
MODEL = "openai/gpt-oss-120b"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DB_PATH = os.getenv("DB_PATH", "neuron.db")

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("neuron")
app = Flask(__name__)
http = requests.Session()


def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with db() as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL
        )""")
        conn.execute("""CREATE TABLE IF NOT EXISTS knowledge (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL, answer TEXT NOT NULL, created_by INTEGER
        )""")


def tg(method, data=None, timeout=30):
    r = http.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/{method}",
                  json=data or {}, timeout=timeout)
    r.raise_for_status()
    result = r.json()
    if not result.get("ok"):
        raise RuntimeError(result.get("description", "Telegram API error"))
    return result.get("result")


def send(chat_id, text):
    tg("sendMessage", {"chat_id": chat_id, "text": str(text)[:4000]})


def owner(user_id):
    return OWNER_ID is not None and int(user_id) == OWNER_ID


def history(user_id):
    with db() as conn:
        rows = conn.execute(
            "SELECT role,content FROM history WHERE user_id=? ORDER BY id DESC LIMIT 20",
            (int(user_id),)).fetchall()
    return [{"role": x["role"], "content": x["content"]} for x in reversed(rows)]


def save(user_id, role, content):
    with db() as conn:
        conn.execute("INSERT INTO history(user_id,role,content) VALUES(?,?,?)",
                     (int(user_id), role, content))
        conn.execute("""DELETE FROM history WHERE user_id=? AND id NOT IN
            (SELECT id FROM history WHERE user_id=? ORDER BY id DESC LIMIT 20)""",
                     (int(user_id), int(user_id)))


def knowledge():
    with db() as conn:
        return conn.execute("SELECT question,answer FROM knowledge ORDER BY id DESC LIMIT 25").fetchall()


def ask_ai(user_id, prompt):
    facts = knowledge()
    extra = ""
    if facts:
        extra = "\n\nЗнания от создателя:\n" + "\n".join(
            f"Вопрос: {x['question']}\nОтвет: {x['answer']}" for x in facts)
    messages = [{"role": "system", "content":
        "Ты НЕЙРОН — дружелюбный и умный русскоязычный ИИ-помощник. "
        "Отвечай естественно, учитывай историю, не выдумывай факты." + extra}]
    messages.extend(history(user_id))
    messages.append({"role": "user", "content": prompt})
    r = http.post(GROQ_URL,
        headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
        json={"model": MODEL, "messages": messages, "temperature": 0.8, "max_tokens": 1400},
        timeout=75)
    if r.status_code != 200:
        try:
            detail = r.json().get("error", {}).get("message", r.text)
        except Exception:
            detail = r.text
        raise RuntimeError(f"Groq API {r.status_code}: {detail}")
    answer = r.json()["choices"][0]["message"]["content"].strip()
    save(user_id, "user", prompt)
    save(user_id, "assistant", answer)
    return answer or "Не получилось сформировать ответ. Попробуй ещё раз."


def handle(update):
    message = update.get("message")
    if not message:
        return
    chat_id = message.get("chat", {}).get("id")
    user_id = message.get("from", {}).get("id")
    text = (message.get("text") or "").strip()
    if not chat_id or not user_id or not text:
        return

    if text.startswith("/start"):
        send(chat_id, "Привет! Я НЕЙРОН — ИИ-помощник. Просто напиши мне сообщение. Команды: /help, /id")
    elif text.startswith("/help"):
        send(chat_id, "/start — начало\n/id — твой Telegram ID\n/reset — очистить историю\n/teach вопрос | ответ — обучить (создатель)\n/knowledge — база знаний (создатель)")
    elif text.startswith("/id"):
        send(chat_id, f"Твой Telegram ID: {user_id}")
    elif text.startswith("/reset"):
        with db() as conn:
            conn.execute("DELETE FROM history WHERE user_id=?", (int(user_id),))
        send(chat_id, "История очищена.")
    elif text.startswith("/teach"):
        if not owner(user_id):
            send(chat_id, "Команда доступна только создателю.")
            return
        raw = text[len("/teach"):].strip()
        if "|" not in raw:
            send(chat_id, "Формат: /teach вопрос | ответ")
            return
        q, a = [v.strip() for v in raw.split("|", 1)]
        if not q or not a:
            send(chat_id, "Нужны и вопрос, и ответ.")
            return
        with db() as conn:
            conn.execute("INSERT INTO knowledge(question,answer,created_by) VALUES(?,?,?)",
                         (q, a, int(user_id)))
        send(chat_id, "Добавил в общую базу знаний.")
    elif text.startswith("/knowledge"):
        if not owner(user_id):
            send(chat_id, "Команда доступна только создателю.")
            return
        rows = knowledge()
        send(chat_id, "\n".join(f"{i}. {x['question']} → {x['answer']}" for i, x in enumerate(rows, 1))
             if rows else "База знаний пока пустая.")
    elif text.startswith("/clear_knowledge"):
        if not owner(user_id):
            send(chat_id, "Команда доступна только создателю.")
            return
        with db() as conn:
            conn.execute("DELETE FROM knowledge")
        send(chat_id, "База знаний очищена.")
    else:
        try:
            tg("sendChatAction", {"chat_id": chat_id, "action": "typing"})
            send(chat_id, ask_ai(user_id, text))
        except Exception as e:
            log.exception("Message handling error")
            send(chat_id, f"Ошибка ИИ: {str(e)[:700]}")


@app.get("/")
def index():
    return "НЕЙРОН работает", 200


@app.post("/webhook")
def webhook():
    if WEBHOOK_SECRET and request.headers.get("X-Telegram-Bot-Api-Secret-Token") != WEBHOOK_SECRET:
        abort(403)
    update = request.get_json(silent=True) or {}
    try:
        handle(update)
    except Exception:
        log.exception("Webhook update failed")
    return "ok", 200


def setup():
    if not TELEGRAM_TOKEN or not GROQ_API_KEY:
        log.error("Set TELEGRAM_TOKEN and GROQ_API_KEY in hosting environment variables.")
        return
    init_db()
    base = os.getenv("RENDER_EXTERNAL_URL", "").rstrip("/")
    if base:
        url = base + "/webhook"
        data = {"url": url}
        if WEBHOOK_SECRET:
            data["secret_token"] = WEBHOOK_SECRET
        try:
            tg("setWebhook", data)
            log.info("Telegram webhook configured: %s", url)
        except Exception:
            log.exception("Could not configure Telegram webhook")


setup()
