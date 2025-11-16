from pprint import pprint

from ollama import chat

MODEL = "tinyllama"

response_stream = chat(
    model=MODEL,
    messages=[
        {"role": "user", "content": "Why is the sky blue? In one sentence."},
    ],
    stream=True,
)

pprint(response_stream)
type(response_stream)  # generator
# generator (iterator)

for chunk in response_stream:
    print(
        chunk.message.content,
        # end="\n",
        # flush=True,
    )
