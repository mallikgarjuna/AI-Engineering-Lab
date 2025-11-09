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


# Text summarization ==================
def summarizing_long_text():
    """Abstractive summarization."""
    from transformers import pipeline

    original_text = """Greece has many islands, with estimates ranging from somewhere around 1,200 to 6,000, depending on the minimum size to take into account. The number of inhabited islands is variously cited as between 166 and 227.
    The Greek islands are traditionally grouped into the following clusters: the Argo-Saronic Islands in the Saronic Gulf near Athens; the Cyclades, a large but dense collection occupying the central part of the Aegean Sea; the North Aegean islands, a loose grouping off the west coast of Turkey; the Dodecanese, another loose collection in the southeast between Crete and Turkey; the Sporades, a small tight group off the coast of Euboea; and the Ionian Islands, chiefly located to the west of the mainland in the Ionian Sea. Crete with its surrounding islets and Euboea are traditionally excluded from this grouping.
    """

    summarizer = pipeline(task="summarization", model="cnicu/t5-small-booksum")

    summary_text = summarizer(original_text)

    print(f"Original text length: {len(original_text)}")
    print(f"Original text length: {len(summary_text[0]['summary_text'])}")
    # Original text length: 836
    # Original text length: 473


# The summary is shorter and demonstrates how abstractive summarization generates new, concise sentences. Unlike extractive methods, it rephrases content. Next, let's explore the minimum and maximum length parameters of the summarization pipeline!


def adjusting_summary_length():
    from transformers import pipeline

    original_text = "Greece has many islands, with estimates ranging from somewhere around 1,200 to 6,000, depending on the minimum size to take into account. The number of inhabited islands is variously cited as between 166 and 227. The Greek islands are traditionally grouped into the following clusters: the Argo-Saronic Islands in the Saronic Gulf near Athens; the Cyclades, a large but dense collection occupying the central part of the Aegean Sea; the North Aegean islands, a loose grouping off the west coast of Turkey; the Dodecanese, another loose collection in the southeast between Crete and Turkey; the Sporades, a small tight group off the coast of Euboea; and the Ionian Islands, chiefly located to the west of the mainland in the Ionian Sea. Crete with its surrounding islets and Euboea are traditionally excluded from this grouping."

    # Generate a summary of original_text between 1 and 10 tokens
    short_summarizer = pipeline(
        task="summarization",
        model="cnicu/t5-small-booksum",
        min_new_tokens=1,
        max_new_tokens=10,
    )

    short_summary_text = short_summarizer(original_text)

    print(short_summary_text[0]["summary_text"])


# Auto Models and Tokenizers =============
def tokenizing_text_with_AutoTokenizer():
    from transformers import AutoTokenizer

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "distilbert-base-uncased-finetuned-sst-2-english"
    )

    # Split input text into tokens
    tokens = tokenizer.tokenize("AI: Making robots smarter and human lazier!")

    # Display the tokenized output
    print(f"Tokenzied output: {tokens}")
    # ['ai', ':', 'making', 'robots', 'smarter', 'and', 'humans', 'la', '##zier', '!']


def using_AutoClasses():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

    # Download the model and tokenizer
    my_model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased-finetuned-sst-2-english"
    )
    my_tokenizer = AutoTokenizer.from_pretrained(
        "distilbert-base-uncased-finetuned-sst-2-english"
    )

    # Create the pipeline
    my_pipeline = pipeline(
        task="sentiment-analysis", model=my_model, tokenizer=my_tokenizer
    )

    # Predict the sentiment
    output = my_pipeline("This course is pretty good, I guess.")
    print(f"Sentiment using AutoClasses: {output[0]['label']}")


if __name__ == "__main__":
    # grammatical_correctness()
    # qnli_pipeline()
    # dynamic_category_assignment()
    # summarizing_long_text()
    # adjusting_summary_length()  # didn't run this
    tokenizing_text_with_AutoTokenizer()  # didn't run this
    using_AutoClasses()
