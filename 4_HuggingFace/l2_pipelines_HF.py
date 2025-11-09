import os

from dotenv import load_dotenv

load_dotenv()


# Text Classification ============
def grammatical_correctness():
    from transformers import pipeline

    # Create a pipeline for grammar checking
    grammar_checker = pipeline(
        task="text-classification", model="abdulmatinomotoso/English_Grammar_Checker"
    )

    # Check grammar of the input text
    output = grammar_checker("I will walk dog")
    print(output)
    # [{'label': 'LABEL_0', 'score': 0.9956323504447937}]


# The grammar_checker pipeline classified the input text as "LABEL_0", or unacceptable grammar. Now, let's move on to Question Natural Language Inference!


def qnli_pipeline():
    from transformers import pipeline

    classifier = pipeline(
        task="text-classification",
        model="cross-encoder/qnli-electra-base",
    )

    output = classifier(
        inputs="Where is the capital of France?, Brittany is known for its stunning coastline.",
    )

    print(output)
    # [{'label': 'LABEL_0', 'score': 0.016211949288845062}]


# the QNLI pipeline showed that the text could not answer the question with a low score. Let's move on to dynamic category assignment.


def dynamic_category_assignment():
    from transformers import pipeline

    text = "AI-powered robots assist in complex brain surgeries with precision."

    classifier = pipeline(
        task="zero-shot-classification",
        model="facebook/bart-large-mnli",  # >1.5GB
    )

    categories = ["politics", "science", "sports"]

    output = classifier(text, categories)

    print(f"Top Label: {output['labels'][0]} with score: {output['scores'][0]}")
    # Top Label: science with score: 0.9510334134101868


# You were able to use the model to predict brand new classes! This was just the beginning. There is so much more to explore in text classification pipelines with Hugging Face.


# Text summarization

if __name__ == "__main__":
    # grammatical_correctness()
    # qnli_pipeline()
    dynamic_category_assignment()
