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
        {"role": "user", "content": "What's the capital of India? Be short."},
    ],
}


# 1. Simple/regular response with HTTPX.post() ===================
def http_response_simple() -> None:
    # Be default, this endpoint `/api/chat` gives streaming response
    # To disable it, use `stream=False` in payload;
    # payload_without_stream = payload.update(stream=False)
    response_simple = httpx.post(url=url, json={**payload, "stream": False})
    # data = response_simple.json() # data is a plain Python dict
    # data = response_simple.json() # .json() method of the HTTPX Response object
    data = json.loads(response_simple.text)  # Same as above .json()

    # This doesn't work: b/c data is a dict, not a Py object
    # print(data.message.content)

    # This works - b/c data is a dict
    print("Assistant reply WITHOUT streaming:")
    print(data["message"]["content"])


# 2. Stream response with HTTPX.stream() ===================
# Send the HTTP POST request with streaming enabled using HTTPX.stream()
# response = httpx.post(url=url, data=data) # This doesn't work for streaming
def http_response_stream() -> None:
    try:  # For handling request error
        with httpx.stream(method="POST", url=url, json=payload) as response_stream:
            # print(response.status_code)
            try:  # For handling response error
                response_stream.raise_for_status()
            except httpx.HTTPStatusError as exc:
                print(
                    f"Error response {exc.response.status_code} while requesting {exc.request.url!r}"
                )

            print("\nAssistant reply WITH streaming:")
            for line in response_stream.iter_lines():
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


if __name__ == "__main__":
    http_response_simple()

    http_response_stream()
