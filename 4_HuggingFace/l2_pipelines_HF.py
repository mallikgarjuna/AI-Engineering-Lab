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


if __name__ == "__main__":
    grammatical_correctness()
    # qnli_pipeline()
