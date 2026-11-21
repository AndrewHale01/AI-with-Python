import json
import os
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")


def generate_response(messages):
    payload = json.dumps(
        {"model": MODEL, "messages": messages, "stream": False}
    ).encode("utf-8")
    request = Request(
        OLLAMA_CHAT_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urlopen(request, timeout=120) as response:
        result = json.loads(response.read().decode("utf-8"))

    return result["message"]["content"]


def main():
    print(f"Локальний чат-бот запущено (модель: {MODEL}).")
    print("Введіть повідомлення. Для завершення введіть Bye.")
    messages = []

    while True:
        try:
            user_message = input("Ви: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nЧат завершено.")
            return 0

        if user_message.lower() == "bye":
            print("Бот: До побачення!")
            return 0
        if not user_message:
            continue

        messages.append({"role": "user", "content": user_message})
        try:
            answer = generate_response(messages)
        except HTTPError as error:
            details = error.read().decode("utf-8", errors="replace")
            print(
                f"Помилка Ollama (HTTP {error.code}): {details}",
                file=sys.stderr,
            )
            return 1
        except URLError as error:
            print(
                "Не вдалося підключитися до Ollama. Переконайтеся, "
                "що Ollama запущена та модель завантажена "
                f"(ollama pull {MODEL}). Деталі: {error.reason}",
                file=sys.stderr,
            )
            return 1

        messages.append({"role": "assistant", "content": answer})
        print(f"Бот: {answer}")


if __name__ == "__main__":
    raise SystemExit(main())