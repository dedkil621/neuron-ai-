# -*- coding: utf-8 -*-
"""НЕЙРОН — Telegram AI bot: chat, photo understanding and image generation."""

import os
import time
import base64
import sqlite3
import logging
import mimetypes
from urllib.parse import quote

import requests
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
POLLINATIONS_API_KEY = os.getenv("POLLINATIONS_API_KEY", "").strip()
OWNER_ID_RAW = os.getenv("OWNER_ID", "").strip()
OWNER_ID = int(OWNER_ID_RAW) if OWNER_ID_RAW.isdigit() else None

TEXT_MODEL = "openai/gpt-oss-120b"
VISION_MODEL = "qwen/qwen3.6-27b"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
POLLINATIONS_URL = "https://gen.pollinations.ai/image/"
DB_PATH = os.getenv("DB_PATH", "neuron.db")

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(message)s"
)
log = logging.getLogger("neuron")
http = requests.Session()
http.headers.update({"User-Agent": "NEIRON-Telegram-Bot/2.0"})


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


def send_photo(chat_id, image_bytes, filename="neuron.png", caption=None):
    files = {"photo": (filename, image_bytes, "image/jpeg")}
    data = {"chat_id": chat_id}
    if caption:
        data["caption"] = str(caption)[:900]
    response = http.post(
        f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto",
        data=data,
        files=files,
        timeout=90
    )
    response.raise_for_status()
    payload = response.json()
    if not payload.get("ok"):
        raise RuntimeError(payload.get("description", "Telegram не смог отправить картинку"))
    return payload.get("result")


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
            (int(user_id), role, str(content)[:12000])
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


def system_prompt():
    facts = get_knowledge()
    extra = ""
    if facts:
        extra = "\n\nЗнания, добавленные создателем:\n" + "\n".join(
            f"Вопрос: {row['question']}\nОтвет: {row['answer']}" for row in facts
        )
    return (
        "Ты НЕЙРОН — умный русскоязычный ИИ-помощник. "
        "Общайся естественно, без канцелярита и шаблонных повторов. "
        "Учитывай историю диалога, отвечай по существу и не выдумывай факты. "
        "Если не знаешь ответа, честно скажи об этом." + extra
    )


def call_groq(user_id, prompt, image_data_url=None):
    if not GROQ_API_KEY:
        raise RuntimeError("Не задан GROQ_API_KEY в файле .env")

    messages = [{"role": "system", "content": system_prompt()}]
    messages.extend(get_history(user_id))

    if image_data_url:
        content = [
            {"type": "text", "text": prompt or "Подробно опиши, что изображено на фотографии."},
            {"type": "image_url", "image_url": {"url": image_data_url}}
        ]
        messages.append({"role": "user", "content": content})
        model = VISION_MODEL
    else:
        messages.append({"role": "user", "content": prompt})
        model = TEXT_MODEL

    response = http.post(
        GROQ_URL,
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 1400
        },
        timeout=120
    )

    if response.status_code != 200:
        try:
            detail = response.json().get("error", {}).get("message", response.text)
        except Exception:
            detail = response.text
        raise RuntimeError(f"Groq API {response.status_code}: {detail[:450]}")

    answer = response.json()["choices"][0]["message"]["content"]
    if isinstance(answer, list):
        answer = " ".join(
            part.get("text", "") for part in answer if isinstance(part, dict)
        )
    answer = str(answer or "").strip()
    if not answer:
        answer = "Не получилось сформировать ответ. Попробуй ещё раз."

    saved_prompt = prompt or "Опиши фотографию."
    save_history(user_id, "user", saved_prompt)
    save_history(user_id, "assistant", answer)
    return answer


def download_photo(file_id):
    file_info = telegram("getFile", {"file_id": file_id})
    file_path = file_info.get("file_path")
    if not file_path:
        raise RuntimeError("Telegram не вернул файл фотографии.")

    url = f"https://api.telegram.org/file/bot{TELEGRAM_TOKEN}/{file_path}"
    response = http.get(url, timeout=60)
    response.raise_for_status()
    if len(response.content) > 18 * 1024 * 1024:
        raise RuntimeError("Фотография слишком большая. Отправь фото поменьше.")
    mime = mimetypes.guess_type(file_path)[0] or "image/jpeg"
    if mime not in ("image/jpeg", "image/png", "image/webp"):
        mime = "image/jpeg"
    encoded = base64.b64encode(response.content).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def generate_image(prompt):
    if not POLLINATIONS_API_KEY:
        raise RuntimeError("Не задан POLLINATIONS_API_KEY в .env. Добавь ключ Pollinations.")
    if not prompt.strip():
        raise RuntimeError("Напиши, какую картинку нужно нарисовать.")

    url = POLLINATIONS_URL + quote(prompt.strip(), safe="")
    response = http.get(
        url,
        headers={"Authorization": f"Bearer {POLLINATIONS_API_KEY}"},
        params={"model": "flux"},
        timeout=240
    )
    if response.status_code != 200:
        detail = response.text[:350]
        try:
            detail = response.json().get("error", detail)
        except Exception:
            pass
        raise RuntimeError(f"Pollinations API {response.status_code}: {detail}")
    content_type = response.headers.get("Content-Type", "")
    if not content_type.startswith("image/"):
        raise RuntimeError("Сервис генерации вернул не изображение. Попробуй другой запрос.")
    return response.content


def is_image_request(text):
    lowered = text.lower().strip()
    triggers = (
        "нарисуй", "сгенерируй картинку", "сгенерируй изображение",
        "создай картинку", "создай изображение", "сделай картинку",
        "сделай изображение", "/image"
    )
    return any(lowered.startswith(item) for item in triggers)


def image_prompt(text):
    raw = text.strip()
    if raw.lower().startswith("/image"):
        return raw[len("/image"):].strip()
    for prefix in (
        "сгенерируй изображение", "сгенерируй картинку",
        "создай изображение", "создай картинку",
        "сделай изображение", "сделай картинку", "нарисуй"
    ):
        if raw.lower().startswith(prefix):
            return raw[len(prefix):].strip(" :,-")
    return raw


def handle_message(message):
    chat_id = message.get("chat", {}).get("id")
    user_id = message.get("from", {}).get("id")
    if not chat_id or not user_id:
        return

    text = (message.get("text") or message.get("caption") or "").strip()
    photos = message.get("photo") or []

    if text.startswith("/start"):
        send(
            chat_id,
            "Привет! Я НЕЙРОН — ИИ-помощник.\n\n"
            "Можешь общаться со мной, отправлять фотографии и просить создавать картинки.\n\n"
            "Примеры:\n"
            "• отправь фото и спроси, что на нём\n"
            "• /image космический кот в неоновом городе\n"
            "• нарисуй уютный дом у озера\n\n"
            "Команды: /help, /id, /reset"
        )
    elif text.startswith("/help"):
        send(
            chat_id,
            "/start — начать\n"
            "/help — помощь\n"
            "/id — твой Telegram ID\n"
            "/reset — очистить историю\n"
            "/image описание — создать картинку\n"
            "Отправь фото с подписью или без — НЕЙРОН его изучит.\n"
            "/teach вопрос | ответ — добавить знание (создатель)\n"
            "/knowledge — показать базу знаний (создатель)\n"
            "/clear_knowledge — очистить базу знаний (создатель)"
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
    elif photos:
        try:
            telegram("sendChatAction", {"chat_id": chat_id, "action": "typing"})
            best_photo = max(photos, key=lambda item: item.get("file_size", 0))
            image_data = download_photo(best_photo["file_id"])
            answer = call_groq(
                user_id,
                text or "Подробно опиши, что изображено на фотографии.",
                image_data_url=image_data
            )
            send(chat_id, answer)
        except Exception as error:
            log.exception("Не удалось обработать фотографию")
            send(chat_id, f"Не получилось изучить фото: {str(error)[:500]}")
    elif text and is_image_request(text):
        prompt = image_prompt(text)
        if not prompt:
            send(chat_id, "Напиши описание картинки. Например: /image рыжий кот в космосе.")
            return
        try:
            telegram("sendChatAction", {"chat_id": chat_id, "action": "upload_photo"})
            image_bytes = generate_image(prompt)
            send_photo(chat_id, image_bytes, filename="neuron-generated.jpg", caption="Готово! Картинка от НЕЙРОНа.")
            save_history(user_id, "user", "Создай изображение: " + prompt)
            save_history(user_id, "assistant", "Я создал изображение по запросу пользователя.")
        except Exception as error:
            log.exception("Не удалось создать изображение")
            send(chat_id, f"Не получилось создать картинку: {str(error)[:500]}")
    elif text:
        try:
            telegram("sendChatAction", {"chat_id": chat_id, "action": "typing"})
            send(chat_id, call_groq(user_id, text))
        except Exception as error:
            log.exception("Не удалось обработать сообщение")
            send(chat_id, f"Ошибка ИИ: {str(error)[:700]}")


def main():
    if not TELEGRAM_TOKEN:
        raise SystemExit("Не найден TELEGRAM_TOKEN. Добавь токен в .env.")
    if not GROQ_API_KEY:
        raise SystemExit("Не найден GROQ_API_KEY. Добавь ключ Groq в .env.")

    init_db()
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
