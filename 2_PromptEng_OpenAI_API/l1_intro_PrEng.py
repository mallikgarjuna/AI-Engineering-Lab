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


# You've now had a sneak peek at prompt engineering and how it effectively
# shapes the model's responses. Get ready for more!

# Key principles of prompt engineering ============
# Clear and concise prompts enhance the model's output and ensure successful task completion.


def using_delimited_prompts_with_fstrings():
    # story = """You are a junior developer at a dynamic startup that generates content
    # with the help of AI. The company believes this technology can revolutionize
    # storytelling, and you are excited to be a part of it"""

    story = "In a distant galaxy, there was a brave space explorer named Alex. Alex had spent years traveling through the cosmos, discovering new planets and meeting alien species. One fateful day, while exploring an uncharted asteroid belt, Alex stumbled upon a peculiar object that would change the course of their interstellar journey forever..."

    prompt = f"""Generate a continuation of a story delimited by triple backticks.
    ```{story}```
    """

    response = get_response(prompt)

    print(f"\n Original story: \n: {story}")
    print(f"\n Generated story: \n: {response}")


# Although the generated story smoothly follows the original,
# the prompt was open-ended, and maybe your startup has some requirements to follow.
# You will deal with this in the following exercise!


def building_specific_and_precise_prompts():
    story = "In a distant galaxy, there was a brave space explorer named Alex. Alex had spent years traveling through the cosmos, discovering new planets and meeting alien species. One fateful day, while exploring an uncharted asteroid belt, Alex stumbled upon a peculiar object that would change the course of their interstellar journey forever..."

    # Create a request to complete the story
    prompt = f"""Complete the story given in the triple backticks 
    with only two paragraphs 
    in the style of Shakespeare.
    ```{story}```
    """

    # Get the generated response
    response = get_response(prompt)

    print("\n Original story: \n", story)
    print("\n Generated story: \n", response)


# crafting efficient prompts using prompt engineering principles.
# This ability is crucial in accurately controlling and guiding the language
# model's responses to adapt to the requirements set by you


# Structured outputs and conditional prompts ====================
def generate_table():
    # Create a prompt that generates the table
    prompt = (
        "Generate a table of 10 books that I should read as a science "
        "fiction lover. The table should have columns for Title, Author, and Year."
    )

    # Get the response
    response = get_response(prompt)
    print(response)


def customizing_output_format():
    text = "The sun was setting behind the mountains, casting a warm golden glow across the landscape. Birds were chirping happily, and a gentle breeze rustled the leaves of the trees. It was a perfect evening for a leisurely stroll in the park  "
    # Create the instructions
    instructions = (
        "Determine the language and generate a suitable title for the text "
        "excerpt provided in the triple backticks delimeter."
    )

    # Create the output format
    output_format = f"""Use the following output format for your response.
    Text: <text provided>
    Language: <language of the text>
    Title: <title for the text>
    """

    # Create the final prompt
    prompt = instructions + output_format + f"```{text}```"
    response = get_response(prompt)
    print(response)


# By effectively designing prompts to customize the output structure based on the
# given input, you've showcased your proficiency in tailoring AI responses to
# specific requirements.


def using_conditional_prompts():
    text = (
        "The sun was setting behind the mountains, "
        "casting a warm golden glow across the landscape.  "
    )
    # Create the instructions
    instructions = (
        "Infer the language, the number of sentences of the given text "
        "provided in the triple backticks delimeter. If the text contains more than "
        "one sentence, generate a suitable title for it, otherwiser, write 'N/A' "
        "for the title."
    )

    # Create the output format
    output_format = f"""Use the following output format for your response.
    Text: <text provided>
    Language: <language of the text provided>
    Number-of-sentences: <number of sentences in the provided text>
    Title: <title for the text>
    """

    prompt = instructions + output_format + f"```{text}```"
    response = get_response(prompt)
    print(response)


# The model clearly follows your requirements! And since the text you used had
# only one sentence, the model did not generate a title for it
# - thanks to your well-crafted instructions!

if __name__ == "__main__":
    # openai_api_message_roles()
    # print(get_response("What is the capital of France?"))
    # exploring_prompt_engineering()
    # using_delimited_prompts_with_fstrings()
    # building_specific_and_precise_prompts()
    # generate_table()
    # customizing_output_format()
    using_conditional_prompts()
