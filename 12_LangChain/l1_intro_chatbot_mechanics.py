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

if __name__ == "__main__":
    # openai_models_in_langchain()
    huggingface_models_in_langchain()
