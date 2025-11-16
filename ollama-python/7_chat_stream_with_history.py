# 7_chat_stream_with_history.py

import pprint
import sys
from json import JSONDecodeError
from typing import Any

import httpx
import ollama


# Test whether local Ollama server is running
# before running this code
def is_ollama_running() -> bool:
    host = "http://localhost:11434"
    endpoint = "/api/version"
    url = host + endpoint

    try:
        response = httpx.get(url=url)
    except httpx.InvalidURL as exc:
        print(f"Invalid URL error: {exc}")
        return False
    except httpx.RequestError as exc:
        print(f"Server/network error while accessing: {exc.request.url}")
        print("Make sure to run Ollama locally with `ollama serve`")
        return False
    else:
        # try-consider-else (TRY300)
        # https://docs.astral.sh/ruff/rules/try-consider-else/
        try:
            data = response.json()
        except JSONDecodeError as exc:
            print(f"Error while parsing JSON data to py obj: {exc.msg}")
        else:
            print(
                f"Ollama version: {data['version']} network response: {response.status_code}",
            )
        return response.status_code == httpx.codes.OK  # instead of hardcoding 200;


# Chat stream
def chat_stream_with_memory() -> None:
    model = "tinyllama"
    message_history: list[dict[str, Any]] = []

    while True:
        user_input = input("User input: ")
        if user_input.lower() in ["exit", "stop", "quit"]:
            print("Chat ending...")
            break

        user_message = {
            "role": "user",
            "content": user_input,
        }

        message_history += [user_message]

        response_chat_stream = ollama.chat(
            model=model,
            messages=message_history,
            stream=True,
        )

        assistant_reply = ""

        for chunk in response_chat_stream:
            if not chunk.message.content:
                continue

            assistant_reply += chunk.message.content
            print(chunk.message.content, end="", flush=True)
        print()

        assistant_message = {
            "role": "assistant",
            "content": assistant_reply,
        }

        message_history += [assistant_message]
        pprint.pprint(message_history, sort_dicts=False)


if __name__ == "__main__":
    if not is_ollama_running():
        sys.exit()
    else:
        chat_stream_with_memory()
