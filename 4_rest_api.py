import json

import httpx

# No need to import ollama
# But run `ollama serve` in local terminal b4 running this module
# import ollama

# Set up the base URL for the local Ollama REST API
# /api/generate for text-only models
# /api/chat for chat-style models
url = "http://localhost:11434/api/chat"

MODEL = "tinyllama"

# Define the payload (user input prompt)
payload = {
    "model": MODEL,
    "messages": [
        {"role": "user", "content": "What's the capital of India?"},
    ],
}

# Send the HTTP POST request with streaming enabled using HTTPX.stream()
# response = httpx.post(url=url, data=data) # This doesn't work for streaming
with httpx.stream(method="POST", url=url, json=payload) as response:
    print(response.status_code)
    for line in response.iter_lines():
        # print(line)
        # Ignore empty lines
        if not line:
            continue

        try:
            # Parse each line as a JSON object
            json_data = json.loads(line)
            # Use 'flush=True' to force the stream output live
            print(json_data["message"]["content"], end="", flush=True)
        except json.JSONDecodeError as exc:
            print(f"Failed to parse line: {line} with error {exc.msg}")
