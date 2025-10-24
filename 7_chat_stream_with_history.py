# 7_chat_stream_with_history.py

import pprint
from typing import Any

import ollama

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
