import os

from dotenv import load_dotenv

load_dotenv()


def openai_models_in_langchain():
    # Install: uv add langchain
    # Install: uv add langchain-openai
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini",
        max_completion_tokens=100,
    )

    prompt = "Three reasons for using LangChain for LLM application development."

    response = llm.invoke(input=prompt)

    print(response.content)


# The standardized syntax that LangChain offers means that models can be
# quickly changed in and out as new ones are released and tested.
# Let's see this in action by trying a model from Hugging Face!


def huggingface_models_in_langchain():
    # Install: uv add langchain-huggingface
    from langchain_huggingface import HuggingFacePipeline

    # Install: transformers to use this: 'uv add transformers'
    # Install: uv add torch - to install PyTorch
    # Please note that you may need to restart your runtime after installation.
    llm = HuggingFacePipeline.from_model_id(
        model_id="crumb/nano-mistral",
        task="text-generation",
        pipeline_kwargs={"max_new_tokens": 20},
    )

    prompt = "Hugging Face is"

    response = llm.invoke(input=prompt)

    print(response)


# This syntax for defining and invoking LLMs unlocks the ability to use almost
# any model! Now that you have nailed the defining and invoking LLM workflow,
# let's look at strategies for prompting these models effectively.


# Prompt Templates:
# Prompt templates are created using LangChain's PromptTemplate class.
# PromptTemplate.from_template(template=template)
# ChatPromptTemplate.from_messages(messages=[(),()])


def prompt_templates_and_chaining():
    from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
    from langchain_openai import ChatOpenAI

    template = (
        "You are an artificial intelligence assistant, answer the questions. {question}"
    )
    prompt_template = PromptTemplate.from_template(template=template)

    llm = ChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini",
        max_completion_tokens=100,
    )

    llm_chain = prompt_template | llm

    question = "How does LangChain make LLM application development easier?"
    response = llm_chain.invoke(input={"question": question})

    print(response.content)


# perfectly pieced-together prompt produced pristine performance!
# Well-designed prompt templates are at the heart of many LangChain applications;
# keep experimenting to find the template that gives you optimum performance!


def chat_prompt_templates():
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI

    prompt_template = ChatPromptTemplate.from_messages(
        messages=[
            (
                "system",
                "You are a geography expert that returns the colors present in a country's flga.",
            ),
            ("human", "France"),
            ("ai", "blue, white, red"),
            ("human", "{country}"),
        ],
    )

    llm = ChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini",
    )

    llm_chain = prompt_template | llm

    # country = "Japan"
    country = "Spain"

    response = llm_chain.invoke(input={"country": country})

    print(response.content)


# Hopefully you're beginning to the power of prompt templates for translating
# user inputs into model inputs. Try experimenting with other countries to see
# how the model performs for more complex flags, like 'Spain'. When you're done,
# head on over to the next video to learn how to scale-up your prompts for larger
# numbers of examples!


# PromptTemplate and ChatPromptTemplate are great for integrating variables,
# but struggle with integrating datasets containing many examples.
# This is where FewShotPromptTemplate comes in!
def creating_few_shot_example_set():
    # create the examples list of dicts
    examples = [
        {
            "question": "How many DataCamp courses has Jack completed?",
            "answer": "36",
        },
        {
            "question": "How much XP does Jack have on DataCamp?",
            "answer": "284,320XP",
        },
        {
            "question": "What technology does Jack learn about most on DataCamp?",
            "answer": "Python",
        },
    ]

    # print(f"Examples set: {examples}")

    return examples


# With your examples dataset all set up,
# you're ready to create your few-shot prompt template!


def building_few_shot_prompt_template():
    from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

    example_promot_template_format = PromptTemplate.from_template(
        template="Question: {question}\n{answer}",
    )

    examples = creating_few_shot_example_set()

    few_shot_prompt_template = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_promot_template_format,
        suffix="Question: {input}",
        input_variables=["input"],
    )

    # This '.invoke()' is for invoking the prompt
    # - different from invoking the llm (see previous examples, above)
    prompt = few_shot_prompt_template.invoke(
        input={"input", "What is Jack's favorite technology on DataCamp?"}
    )

    print(prompt.text)

    return prompt


# Invoking the prompt template allows you to see exactly what context the model
# will have. Now for the final piece: create an LCEL chain to combine the
# few-shot template with an LLM!


def implementing_few_shot_prompting():
    from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
    from langchain_openai import ChatOpenAI

    # Get examples: list of dicts with "question" and "answer" keys
    examples = creating_few_shot_example_set()

    # Create example_prompt template with "question" and "answer"
    example_prompt = PromptTemplate.from_template(
        template="Question: {question}\n{answer}",
    )

    # Create FewShotPromptTemplate
    prompt_template = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        suffix="Question: {input}",
        input_variables=["input"],
    )

    # Instantiate an OpenAI chat LLM
    llm = ChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini",
    )

    # Create the chain
    llm_chain = prompt_template | llm

    response = llm_chain.invoke(
        input={"input": "What is Jack's favorite technology on DataCamp?"},
    )

    print(response.content)


# Being able to integrate external data into prompts is an incredibly valuable skill
# for designing LLM applications like chatbots.

# In the next chapter, you'll continue your LangChain journey to look a different
# types of chains and agents, which allow LLMs to make decisions!

if __name__ == "__main__":
    # openai_models_in_langchain()
    # huggingface_models_in_langchain()
    # prompt_templates_and_chaining()
    # chat_prompt_templates()
    # creating_few_shot_example_set()
    # building_few_shot_prompt_template()
    implementing_few_shot_prompting()
