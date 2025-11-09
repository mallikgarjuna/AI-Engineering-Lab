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


if __name__ == "__main__":
    # grammatical_correctness()
    qnli_pipeline()
