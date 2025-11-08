# Time to level up your LangChain chains!
# You'll learn to use the LangChain Expression Language (LCEL) for defining chains
# with greater flexibility.
# You'll create sequential chains, where inputs are passed between components
# to create more advanced applications.
# You'll also begin to integrate agents, which use LLMs for decision-making.

# Sequential problems


import os

from dotenv import load_dotenv

load_dotenv()


def building_prompt_for_sequential_chains():
    from langchain_core.prompts import PromptTemplate

    # Prompt template that takes an input acitivity
    learning_prompt = PromptTemplate(
        input_variables=["activity"],
        template="I want to learn how to {activity}. Can you suggest how I can learn this step-by-step?",
    )

    # Prompt template that places a time constraint on the output
    time_prompt = PromptTemplate(
        input_variables=["learning_plan"],
        template="I only have one week. Can you create a plan to help me hit this goal: {learning_plan}",
    )

    # Invoke the learning_prompt with an activity
    prompt = learning_prompt.invoke(input={"activity": "play golf"})

    print(prompt)


# With these prompts set up, you're now to set up the LLM and chain them
# all together using LCEL.
# LECL = LangChain Expression Language


def sequential_chains_with_LCEL():
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import PromptTemplate
    from langchain_openai import ChatOpenAI

    learning_activity_template = PromptTemplate(
        template="I want to learn how to {activity}. Can you suggest how I can learn this step-by-step?",
        input_variables=["activity"],
    )

    time_plan_template = PromptTemplate(
        template="I only have one week. Can you create a concise plan to help me hit this goal: {learning_plan}",
        input_variables=["learning_plan"],
    )

    llm = ChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini",
        max_completion_tokens=200,
    )

    # Complete the sequential chain with LCEL:
    # llm → generates an AIMessage
    # StrOutputParser() → extracts the .content as string
    # Without StrOutputParser(): You’d get an AIMessage object — not a string, breaking the next prompt.
    seq_chain = (
        {"learning_plan": learning_activity_template | llm | StrOutputParser()}
        | time_plan_template
        | llm
        | StrOutputParser()
    )

    # Call the chain
    response = seq_chain.invoke(input={"activity": "play the piano"})

    print(response)


# Running a series of chains opens up lots of possibility for designing more
# sophisticated workflows. Now you're beginning to grasp how LangChain can
# handle more sophisticated workflows, it's time to talk about agents,
# which enable LLMs to make decisions.

if __name__ == "__main__":
    # building_prompt_for_sequential_chains()
    sequential_chains_with_LCEL()
