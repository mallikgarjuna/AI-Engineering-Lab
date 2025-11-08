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


# Adding guardrails in the system prompt is a good first line of defence
# against misuse. We'll now guide model outputs in a slightly different way
# using the 'assistant' role.

# Utilizing the assistant role:
# Assistant messages are primarily used for providing examples to the model
# We can experiment with how many examples are necessary to achieve our goals

# Structuring chat messages:
# [{system}, {user}, {assistant}, {user}]
# This system > user-assistant pair examples > new user prompt structure
# is a best practice for sending chat messages.

# Where to provide examples?
# System -> important template formatting
# Assistant -> example conversations
# User -> content required for the new input (often single-turn)


def adding_assistant_messages():
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful Geography tutor that generates concise summaries for different countries.",
            },
            {"role": "user", "content": "Give me a quick summary of Portugal."},
            {
                "role": "assistant",
                "content": "Portugal is a country in Europe that borders Spain. The capital city is Lisboa",
            },
            {"role": "user", "content": "Give me a quick summary of Greece"},
        ],
        max_completion_tokens=100,
    )

    print(response.choices[0].message.content)
    # In this case, one example doesn't seems to help the model decide
    # what sort of information to include, but there's slight deviation in content
    # and length. Let's try adding more messages,
    # going from one-shot to few-shot, to see if this improves!


def adding_more_assistant_messages():
    example1 = "Give me a quick summary of Peru."
    response1 = "Peru is a country in South America that borders Ecuador, Colombia, Brazil, Bolivia, and Chile. The capital city is Lima."
    example2 = "Give me a quick summary of Australia."
    response2 = "Give me a quick summary of Australia."
    example3 = "Give me a quick summary of Canada."
    response3 = "Canada is a country in North America that borders the United States. The capital city is Ottawa."

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful Geography tutor that generates concise summaries for different countries.",
            },
            {"role": "user", "content": "Give me a quick summary of Portugal."},
            {
                "role": "assistant",
                "content": "Portugal is a country in Europe that borders Spain. The capital city is Lisboa",
            },
            {"role": "user", "content": example1},
            {"role": "assistant", "content": response1},
            {"role": "user", "content": example2},
            {"role": "assistant", "content": response2},
            {"role": "user", "content": example3},
            {"role": "assistant", "content": response3},
            {"role": "user", "content": "Give me a quick summary of Greece"},
        ],
        max_completion_tokens=100,
    )

    print(response.choices[0].message.content)


# Providing examples helps mitigate undesirable or inaccurate model responses,
# so they're a good practice in many use cases. Here, however, you can see that
# adding more examples didn't really improve the response length. In this case,
# an example in the system message and a few user-assistant pairs would likely
# provide more optimal responses. Let's learn to create conversations next;


# Multi-turn conversations with GPT
def creating_conversation_history():
    messages = [
        {
            "role": "system",
            "content": "You are a helpful math tutor that speaks concisely.",
        },
        {"role": "user", "content": "Explain what pi is."},
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_completion_tokens=100,
    )

    assistant_message = {
        "role": "assistant",
        "content": response.choices[0].message.content,
    }

    messages.append(assistant_message)
    print(messages)


# Now that you've proved that messages can be stored, you can now try running
# the message history through a loop, which imitates a real-world conversation
# with new questions and answers.

if __name__ == "__main__":
    # utilizing_system_messages()
    # adding_guardrails()
    # adding_assistant_messages()
    # adding_more_assistant_messages()
    creating_conversation_history()
