import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def openai_api_message_roles():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    conversation_messages = [
        {"role": "system", "content": "You are a helpful event management assistant."},
        {
            "role": "user",
            "content": "What are some good conversation starts at networking events?",
        },
        {"role": "assistant", "content": ""},
    ]
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=conversation_messages,
        max_completion_tokens=100,
    )

    print(response.choices[0].message.content)


# Understanding these roles is vital for crafting prompts that harness
# the OpenAI API's full potential.

if __name__ == "__main__":
    openai_api_message_roles()
