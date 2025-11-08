# 3 Roles in Chat Models:
# You discovered the three main roles—system, user, and assistant
# —that facilitate the chat's flow.
# The system role controls the assistant's behavior,
# the user role provides instructions, and
# the assistant responds accordingly.

# Creating Chat Requests:
# You learned how to make requests to the Chat Completions endpoint by setting up
# a list of dictionaries, each representing a message from one of the roles.
# This method allows for greater customizability of responses.

# Extracting Responses: The process to extract the assistant's response involves
# accessing the choices attribute, subsetting the first element to get the
# Choice object, and then retrieving the content through
# the message and content attributes.

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def utilizing_system_messages():
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a study planning assistant that creates plans for learning new skills.",
            },
            {"role": "user", "content": "I want to learn to speak Dutch."},
        ],
        max_tokens=100,
    )

    # Extract the assistant's text response
    print(response.choices[0].message.content)


# AI models have massively augmented how people approach upskilling and learning;
# in the next exercise, you'll take this up a level to use the model
# for code explanation.


def adding_guardrails():
    sys_msg = """You are a study planning assistant that creates plans for learning
    new skills.
    
    If these skills are non related to languages, return the message:
    
    'Apologies, to focus on languages, we no longer create learning plans on other
    topics.'
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": sys_msg},
            {"role": "user", "content": "Help me learn to AI Engineering."},
        ],
        max_completion_tokens=100,
    )

    print(response.choices[0].message.content)


if __name__ == "__main__":
    # utilizing_system_messages()
    adding_guardrails()
