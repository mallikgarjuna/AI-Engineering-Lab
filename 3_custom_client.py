# https://github.com/ollama/ollama-python

from pprint import pprint

from ollama import Client, ResponseError, pull

ollama_client = Client(
    host="http://localhost:11434",
    # headers={"x-some-header": "some-value"},
)

MODEL = "tinyllama"
# MODEL = "tinyllamaxx" # for testing ResponseError
message = {
    "role": "user",
    "content": "Why is the sky blue? In a short sentence.",
}

try:
    response = ollama_client.chat(
        model=MODEL,
        messages=[message],
    )
except ResponseError as e:
    print("Error: ", e.error)
    print(f"Status code: {e.status_code}")
    if e.status_code == 404:
        pull(model=MODEL)

print(response)
pprint(response)
pprint(response.model_dump())

print(response.message.content)
