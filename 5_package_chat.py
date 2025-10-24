# This uses `ollama` python package
# No need to run Ollama locally using `ollama serve`

import ollama
from rich import print as rich_print

model = "tinyllama"

messages = [
    {
        "role": "user",
        "content": "What is the capital of India?",
    },
]

# Simple chat ==============
# response_chat = ollama.chat(model=model, messages=messages, stream=False)
# # rich_print(response_chat)
# rich_print(response_chat.message.content)

# Steam chat ===============
response_chat_stream = ollama.chat(model=model, messages=messages, stream=True)

for part in response_chat_stream:
    print(part.message.content, end="", flush=True)
