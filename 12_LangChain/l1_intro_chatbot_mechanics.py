import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4o-mini",
    max_completion_tokens=100,
)


def openai_models_in_langchain():
    prompt = "Three reasons for using LangChain for LLM application development."

    response = llm.invoke(input=prompt)

    print(response.content)


if __name__ == "__main__":
    openai_models_in_langchain()
