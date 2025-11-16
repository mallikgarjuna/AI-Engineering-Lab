import pprint

import ollama

model = "tinyllama"

message_history: list[dict] = []

while True:
    user_input = input("Chat with history. Your input: ")
    messages = [
        *message_history,  # Unpack message_history
        {
            "role": "user",
            "content": user_input,
        },
    ]

    response_chat = ollama.chat(model=model, messages=messages, stream=False)

    message_history += [
        {
            "role": "user",
            "content": user_input,
        },
        {
            "role": response_chat.message.role,  # 'assistant'
            "content": response_chat.message.content,
        },
    ]

    print(response_chat.message.content, flush=True)

    pprint.pprint(message_history, sort_dicts=False)
