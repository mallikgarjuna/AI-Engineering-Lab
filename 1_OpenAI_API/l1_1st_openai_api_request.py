import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# Create a request to chat completion endpoint
response = client.chat.completions.create(
    model="gpt-4o-mini",
    max_completion_tokens=100,
    messages=[
        {
            "role": "user",
            "content": "In two sentences, how can the OpenAI be used to upskill myself?",
        },
    ],
)

print(response.choices[0].message.content)

# Use interactive notebook on vscode
# Install pydantic for .model_dump()
response.model_dump()
