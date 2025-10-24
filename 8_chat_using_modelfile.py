import ollama

# Created using Modelfile
# First create this custom model
# using `ollama create mario_model -f ./Modelfile`
model = "mario_model"

message = {
    "role": "user",
    "content": "who are you?",
}

messages = [message]

response_chat_simple = ollama.chat(model=model, messages=messages, stream=False)
data = response_chat_simple.message.content
print(data)
