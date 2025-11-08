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

if __name__ == "__main__":
    # openai_models_in_langchain()
    # huggingface_models_in_langchain()
    # prompt_templates_and_chaining()
    chat_prompt_templates()
