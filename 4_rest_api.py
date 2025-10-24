import json

import httpx
import ollama

# /api/generate for text-only models
# /api/chat for chat-style models
url = "http://localhost:11434/api/chat"

MODEL = "tinyllama"


payload = {
    "model": MODEL,
    "messages": [
        {"role": "user", "content": "What's the capital of India?"},
    ],
}

# response = httpx.post(url=url, data=data)
# response = httpx.post(url=url, json=data)
with httpx.stream(method="POST", url=url, json=payload) as response:
    print(response.status_code)
    for line in response.iter_lines():
        # print(line)
        if not line:
            continue
        json_data = json.loads(line)
        # Use 'flush=True' to force the stream output live
        print(json_data["message"]["content"], end="", flush=True)


# print(response)
# print(response.status_code)
# print(response.text)
# type(response.text)
# {"model":"tinyllama","created_at":"2025-10-23T05:00:39.666745737Z","message":{"role":"assistant","content":"."},"done":false}
# print(response.json())
# print(response.message.content)

# for line in response.text.
