# Hugging Face offers a range of impressive benefits,
# but not every feature is guaranteed or universally
# applicable—performance and suitability depend on specific
# use cases and contexts.

# Hugging Face offers a range of impressive benefits, but not every feature is guaranteed or universally applicable—performance and suitability depend on specific use cases and contexts.
import os

from dotenv import load_dotenv

load_dotenv()


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


def inference_providers():
    """Didn't run this code --- didn't create HF_API_KEY yet!"""
    from huggingface_hub import InferenceClient

    client = InferenceClient(
        provider="together",
        api_key=os.getenv("HF_API_KEY"),
    )

    response = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-V3",
        messages=[{"role": "user", "content": "What is the capital of Belgium?"}],
    )

    print(response.choices[0].message)


# Hugging Face Datasets ======================
def loading_datasets():
    # Install: uv add datasets
    from datasets import load_dataset

    my_dataset = load_dataset(
        "TIGER-Lab/MMLU-Pro",
        split="validation",
    )

    print(my_dataset)


if __name__ == "__main__":
    # building_text_generation_pipeline()
    # inference_providers()
    loading_datasets()
