from pprint import pprint

from ollama import ChatResponse, chat

MODEL = "tinyllama"

response: ChatResponse = chat(
    model=MODEL,
    messages=[{"role": "user", "content": "Why is the sky blue? In one sentence."}],
    stream=False,
)

pprint(response)
pprint(response.model_dump())
for key, value in response.model_dump().items():
    print(key)

type(response)
type(response.message)

pprint(response.model_dump_json())
pprint(response.message.model_dump())
pprint(response.message.model_dump_json())
# print(response)
print(response.message.content)
