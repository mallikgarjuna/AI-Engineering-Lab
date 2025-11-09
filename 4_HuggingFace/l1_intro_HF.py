# Hugging Face offers a range of impressive benefits,
# but not every feature is guaranteed or universally
# applicable—performance and suitability depend on specific
# use cases and contexts.

# Hugging Face offers a range of impressive benefits, but not every feature is guaranteed or universally applicable—performance and suitability depend on specific use cases and contexts.


def building_text_generation_pipeline():
    # transformers was already installed!?
    from transformers import pipeline

    gpt2_pipeline = pipeline(
        task="text-generation",
        model="openai-community/gpt2",
    )

    # Generate three text outputs with a maximum length of 10 tokens
    results = gpt2_pipeline(
        text_inputs="What if AI",
        max_new_tokens=10,
        num_return_sequences=3,
    )

    for result in results:
        print(result["generated_text"])


# You’ve successfully built a text generation pipeline, customized its output, and explored how to generate multiple variations. Next, you’ll explore saving and reloading models!

if __name__ == "__main__":
    building_text_generation_pipeline()
