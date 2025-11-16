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
try:
    response_chat_stream = ollama.chat(model=model, messages=messages, stream=True)

    for chunk in response_chat_stream:
        print(chunk.message.content, end="", flush=True)
except ollama.RequestError as exc:
    rich_print(f"[yellow]An error occured while requesting:[/yellow] {exc.error}")
except ollama.ResponseError as exc:
    rich_print(f"[red]An error occured in response:[/red] {exc.error}")
except Exception as exc:
    rich_print(f"[red]Unexpected error:[/red] {exc}")
