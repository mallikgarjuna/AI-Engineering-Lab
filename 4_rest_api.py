# This is to run Ollama locally
# First need to `ollama serve` before running this module

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
try:  # For handling request error
    with httpx.stream(method="POST", url=url, json=payload) as response:
        # print(response.status_code)
        try:  # For handling response error
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            print(
                f"Error response {exc.response.status_code} while requesting {exc.request.url!r}"
            )

        for line in response.iter_lines():
            # print(line)
            # Ignore empty lines
            if not line:
                continue

            try:
                # Parse each line as a JSON object
                # Take this text line that looks like JSON, and convert (parse) it into a real Python object (a dictionary).
                # B/c: type(line) # <class 'str'>, it's a "JSON string", not a native Python dict obj;
                # This is a JSON string — just text that looks like a dictionary but isn’t one yet.
                json_data = json.loads(line)
                # Now, type(json_data) # <class 'dict'>; so I can safely use dict methods

                # Use 'flush=True' to force the stream output live
                print(json_data["message"]["content"], end="", flush=True)
            except json.JSONDecodeError as exc:
                print(f"Failed to parse line: {line} with error {exc.msg}")
except httpx.RequestError as exc:
    print(f"An error occured while requesting {exc.request.url!r}")
