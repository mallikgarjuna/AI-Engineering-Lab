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
    tools = load_tools(
        tool_names=["wikipedia"],
        llm=llm,
    )

    # Define the agent
    agent = create_agent(
        model=llm,
        tools=tools,
    )

    # Invoke the agent
    response = agent.invoke(
        input={"messages": [("human", "How many people live in New York City?")]}
    )

    print(response)
    print(response["messages"][-1].content)


# The wikipedia tool is pretty handy for quickly looking up specific facts
# to enter into your workflows, potentially overcoming an LLM's knowledge gap.
# Head on over to the final video of the chapter to learn how to create custom tools
# for your agents!


# Custom tools for agents
def defining_function_for_tool_use():
    # Install: uv add pandas

    def retrieve_customer_info(name: str) -> str:
        """Retrieve customer information based on their name."""
        import pandas as pd

        # Define customer data as a list of dicts
        customers_data = [
            {
                "id": 101,
                "name": "Alpha Analytics",
                "subscription_type": "Basic",
                "active_users": 120,
                "auto_renewal": True,
            },
            {
                "id": 102,
                "name": "Blue Horizon Ltd.",
                "subscription_type": "Standard",
                "active_users": 350,
                "auto_renewal": False,
            },
            {
                "id": 103,
                "name": "Crest Dynamics",
                "subscription_type": "Enterprise",
                "active_users": 1200,
                "auto_renewal": True,
            },
            {
                "id": 104,
                "name": "Peak Performance Co.",
                "subscription_type": "Premium",
                "active_users": 800,
                "auto_renewal": True,
            },
            {
                "id": 105,
                "name": "Quantum Systems",
                "subscription_type": "Standard",
                "active_users": 640,
                "auto_renewal": False,
            },
        ]

        # Create the df
        customers_df = pd.DataFrame(data=customers_data)

        # customers_df is not defined in this exercise (but only in course)
        customer_info = customers_df[customers_df["name"] == name]

        return customer_info.to_string()  # return string data

    # Call the function
    result = retrieve_customer_info("Peak Performance Co.")
    print(result)


# you've created a function to load and extract employee data.
# The next step is to create a custom tool for your agent to use.


def creating_custom_tools():
    from langchain_core.tools import tool

    @tool
    def retrieve_customer_info(name: str) -> str:
        """Retrieve customer information based on their name."""
        import pandas as pd

        # Define customer data as a list of dicts
        customers_data = [
            {
                "id": 101,
                "name": "Alpha Analytics",
                "subscription_type": "Basic",
                "active_users": 120,
                "auto_renewal": True,
            },
            {
                "id": 102,
                "name": "Blue Horizon Ltd.",
                "subscription_type": "Standard",
                "active_users": 350,
                "auto_renewal": False,
            },
            {
                "id": 103,
                "name": "Crest Dynamics",
                "subscription_type": "Enterprise",
                "active_users": 1200,
                "auto_renewal": True,
            },
            {
                "id": 104,
                "name": "Peak Performance Co.",
                "subscription_type": "Premium",
                "active_users": 800,
                "auto_renewal": True,
            },
            {
                "id": 105,
                "name": "Quantum Systems",
                "subscription_type": "Standard",
                "active_users": 640,
                "auto_renewal": False,
            },
        ]

        # Create the df
        customers_df = pd.DataFrame(data=customers_data)

        # customers_df is not defined in this exercise (but only in course)
        customer_info = customers_df[customers_df["name"] == name]

        return customer_info.to_string()  # return string data

    print(retrieve_customer_info.name)
    print(retrieve_customer_info.description)
    print(retrieve_customer_info.return_direct)
    print(retrieve_customer_info.args)  # {'name': {'title': 'Name', 'type': 'string'}}


# The @tool decorator is a really nice way to quickly convert Python functions
# into custom tools that are compatible with LangChain agents.
# Time for the final step: head on over to the next exercise
# to integrate this tool with an agent!

if __name__ == "__main__":
    # building_prompt_for_sequential_chains()
    # sequential_chains_with_LCEL()
    # react_agents()
    # defining_function_for_tool_use()
    creating_custom_tools()
