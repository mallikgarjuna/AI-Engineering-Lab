# Time to level up your LangChain chains!
# You'll learn to use the LangChain Expression Language (LCEL) for defining chains
# with greater flexibility.
# You'll create sequential chains, where inputs are passed between components
# to create more advanced applications.
# You'll also begin to integrate agents, which use LLMs for decision-making.

# Sequential problems


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

if __name__ == "__main__":
    building_prompt_for_sequential_chains()
