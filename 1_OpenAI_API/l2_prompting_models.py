# prompt = """""" then use it in "content": prompt;
# OR: text = ".."; prompt = f"....{text}"
# Control response length with tokens
# Cost: based on model and tokens
# - input vs output tokesn
# Always estimate costs before deploying AI features at scale

# Quickly adapting text in this way was impossible before AI
# came along, and now, anyone can do this with just a few
# lines of code and an OpenAI API key.

# Summarization:
# Summarization is one of the most widely-used capabilities
# of OpenAI's completions models. This has many applications
# in business to condense and personalize reports and papers
# to a particular length and audience.

# Find and replace
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

prompt1 = """Replace car with plane and adjust phrase:
A car is a vehicle that is typically powered by an internal combustion engine
or an electric motor. It has four wheels, and is designed to carry passengers
and/or cargo on roads or highways. Cars have become a ubiquitous part of modern
society, and are used for a wide variety of purposes, such as commuting, travel,
and transportation of goods. Cars are often associated with freedom, independence,d
and mobility.
"""

prompt2 = f"""Summarize the following text into two concise bullet points:
{prompt1}
"""

MAX_COMPLETION_TOKENS = 100


def summarize_text():
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": prompt2,
            },
        ],
        max_completion_tokens=100,
    )

    print(response.choices[0].message.content)

    response.model_dump()


def calculate_cost():
    """
    Calculating costs is key to using AI efficiently.

    This task was inexpensive, but at scale or with advanced
    models, costs can add up quickly. Always estimate before
    deploying AI solutions.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": prompt2,
            },
        ],
        max_completion_tokens=MAX_COMPLETION_TOKENS,
    )

    input_token_price = 0.15 / 1_00_000
    output_token_price = 0.6 / 1_00_000

    # Extract token usage
    if response and response.usage:
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
    else:
        print("No 'usage' returned.")

    # Calculate cost
    cost = input_tokens * input_token_price + output_tokens * output_token_price

    print(f"Estimated cost: ${cost}")  # $0.0005235

    # print(response.choices[0].message.content)


if __name__ == "__main__":
    # summarize_text()
    calculate_cost()
