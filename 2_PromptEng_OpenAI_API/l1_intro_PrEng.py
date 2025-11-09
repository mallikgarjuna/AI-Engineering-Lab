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


def get_response(prompt):
    import os

    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Create a request to the chat completions endpoint
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=200,
        temperature=0,
    )

    return response.choices[0].message.content


# You've created a function to make your life easier in the following exercises.
# You'll only need to write one line of code to receive a response from the API.


def exploring_prompt_engineering():
    prompt = (
        "Generate a poem about ChatGPT in basic English that a child can understand."
    )

    response = get_response(prompt)

    print(response)


if __name__ == "__main__":
    # openai_api_message_roles()
    # print(get_response("What is the capital of France?"))
    exploring_prompt_engineering()
