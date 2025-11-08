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


# AGENTS:
# In LangChain, agents use language models to determine actions.
# Agents often use tools, which are functions called by the agent to
# interact with the system.

# Agents can even use chains and other agents as tools!
# In this video, we'll discuss a type of agent called ReAct agents.
# ReAct = Reason(ing) + Act(ing)

# If you ask a ReAct agent:
# it would start by thinking (Reasoning) about the task and which tool to call,
# call that tool using the information, and
# observe the results from the tool call.

# LangGraph:
# To implement agents, we'll be using LangGraph,
# which is branch of the LangChain ecosystem
# specifically for designing agentic systems, or systems including agents.
# Version used in this course: langgraph==0.2.74

# Agents are a fundamental component of so many LangChain applications,
# as they can be used to create flexible and sophisticated workflows.
# Head on over to the next exercise to implement a Zero-Shot ReAct agent!


def react_agents():
    # Install: uv add langchain-community
    from langchain.agents import create_agent
    from langchain_community.agent_toolkits.load_tools import load_tools
    from langchain_openai import ChatOpenAI
    # from langgraph.prebuilt import create_react_agent # deprecated and moved

    # Define llm
    llm = ChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini",
        max_completion_tokens=100,
    )

    # Define the tools
    # Install: uv add wikipedia
    tools = load_tools(["wikipedia"])

    # Define the agent
    agent = create_agent(model=llm, tools=tools)

    # Invoke the agent
    response = agent.invoke(
        input={"messages": [("human", "How many people live in New York City?")]}
    )

    print(response)
    print(response["messages"][-1].content)


if __name__ == "__main__":
    # building_prompt_for_sequential_chains()
    # sequential_chains_with_LCEL()
    react_agents()
