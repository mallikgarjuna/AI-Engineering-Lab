# One limitation of LLMs is that they have a knowledge cut-off due to being
# trained on data up to a certain point.
# In this chapter, you'll learn to create applications that use
# Retrieval Augmented Generation (RAG) to integrate external data with LLMs.
# The RAG workflow contains a few different processes, including splitting data,
# creating and storing the embeddings using a vector database, and
# retrieving the most relevant information for use in the application.
# You'll learn to master the entire workflow!

# RAG development steps - 3 steps:
# Document loader --> Splitting --> Storage + Retrieval
# There are three primary steps to RAG development in LangChain.
# The first is loading the documents into LangChain with document loaders.
# Next, is splitting the documents into chunks.
# Chunks are units of information that we can index and process individually.
# The last step is encoding and storing the chunks for retrieval,
# which could utilize a vector database if that meets the needs of the use case.

# 1. Integrating Document Loaders
# 2. PDF document loaders


def PDF_document_loaders():
    # Install: uv add pypdf
    from pathlib import Path

    from langchain_community.document_loaders import PyPDFLoader

    # Create document loader
    pdf_file_path = Path(__file__).absolute().parent / "rag_vs_fine_tuning_arxiv.pdf"
    loader = PyPDFLoader(file_path=pdf_file_path)

    # Load the document
    data = loader.load()

    print(data[0])


# Perfectly parsed PDF pages! Imagine where this could lead—you could load a
# library of research papers and use RAG to create a tutor chatbot that helps
# you stay up-to-date on the latest AI research!


def CSV_document_loaders():
    from pathlib import Path

    from langchain_community.document_loaders import CSVLoader

    # This is same as above - same but from specific implementation file
    # while above is from package-level namespace (exporeted from __init__.py)
    # from langchain_community.document_loaders.csv_loader import CSVLoader

    # Data from
    # https://github.com/fivethirtyeight/data/blob/master/fifa/fifa_countries_audience.csv;
    # https://fivethirtyeight.com/features/how-to-break-fifa/

    # Create document loader
    csv_file_path = Path(__file__).absolute().parent / "fifa_countries_audience.csv"
    loader = CSVLoader(file_path=csv_file_path)

    data = loader.load()

    print(data[0])


# CSVLoader creates a document for each row in the CSV file,
# which could be used in an LLM application to summarize data and generate reports.
# On to the final document loader type!


def HTML_document_loaders():
    from pathlib import Path

    # Install: uv add unstructured
    from langchain_community.document_loaders import UnstructuredHTMLLoader

    html_file_path = (
        Path(__file__).absolute().parent / "white_house_executive_order_nov_2023.html"
    )
    loader = UnstructuredHTMLLoader(file_path=html_file_path)

    data = loader.load()

    # Print the first document
    print(data[0])
    # Print the first document's metadata
    print(data[0].metadata)


# Dexterously digested data! Imagine intergrating this data loader into an
# LLM-powered chatbot; you could scrape HTML from websites and
# have a conversation with a website!
# There are lots of other document loaders out there, and
# I encourage you to explore further in the documentation.


# # Splitting external data for retrieval
def splitting_by_character():
    # Import the character splitter
    from langchain_text_splitters import CharacterTextSplitter

    quote = "Words are flowing out like endless rain into a paper cup, \nthey slither while they pass, \nthey slip away across the universe."

    chunk_size = 24
    chunk_overlap = 10  # chars

    # Create an instance of splitter class
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # Split the string (quote) and print the result
    docs_chunks = splitter.split_text(text=quote)

    print(docs_chunks)
    print([len(doc) for doc in docs_chunks])  # [57, 29, 35]


# Brilliantly broken-down blocks, character by character!
# CharacterTextSplitter is a fairly simple splitting strategy,
# and like in this case, can fail to reduce chunks below the specified chunk_size.
# In the next exercise, you'll take it up a gear and create
# a recursive character splitter!


def recursive_splitting_by_character():
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    quote = "Words are flowing out like endless rain into a paper cup,\nthey slither while they pass,\nthey slip away across the universe."

    chunk_size = 24
    chunk_overlap = 10

    # Create an instance of the splitter class
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # Split the document and print the chunks
    docs_chunks = splitter.split_text(text=quote)
    print(docs_chunks)
    print([len(doc) for doc in docs_chunks])  # [21, 21, 22, 23, 10, 21, 20]


# RecursiveCharacterTextSplitter was able to keep the chunks below chunk_size,
# albeit with a few chunks containing little meaning.
# Take a moment to experiment with different chunk_size and chunk_overlap values,
# running the code each time and interpreting the results.
# Then, head over to the next exercise to split some HTML!


def splitting_HTML():
    from pathlib import Path

    from langchain_community.document_loaders import UnstructuredHTMLLoader
    from langchain_text_splitters import CharacterTextSplitter

    # Load the HTML document into memory
    html_file_path = (
        Path(__file__).absolute().parent / "white_house_executive_order_nov_2023.html"
    )
    loader = UnstructuredHTMLLoader(file_path=html_file_path)
    data = loader.load()

    # Define chunk variables
    chunk_size = 300
    chunk_overlap = 100

    # Split the HTML
    splitter = CharacterTextSplitter(
        separator=".",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    docs_chunks = splitter.split_documents(documents=data)
    print(docs_chunks)


# Play around with the chunk_size and chunk_overlap values to see
# how this affects the output! When you're ready, head on over to the next video
# to bring together everything you've learned in this chapter to implement
# a full RAG workflow!

# ===== RAG storage and retrieval using vector databases ============
# So far: we've covered document loading and splitting,
# Now: we'll round-out the RAG workflow with learning about storing and retrieving
# this information using vector databases.


# Here we use: Chroma - b/c it's lightweight and easy to set up;
# Here, 'documents' - meaning chunks (MG)
def preparing_documents_and_vector_db():
    import os
    from pathlib import Path

    from dotenv import load_dotenv

    # Install: uv add langchain-chroma
    from langchain_chroma import Chroma
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_openai import OpenAIEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    # Load env variables
    load_dotenv()

    # Load the PDF document into data
    pdf_file_path = Path(__file__).absolute().parent / "rag_vs_fine_tuning_arxiv.pdf"
    loader = PyPDFLoader(file_path=pdf_file_path)
    data = loader.load()

    # Split the PDF document's data into chunks
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=300,
        chunk_overlap=50,
    )
    docs_chunks = splitter.split_documents(documents=data)

    # Embed the chunks/documents in a persistent Chroma vector database
    # First define the embeddings model - using OpenAIEmbeddings
    embedding_function = OpenAIEmbeddings(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="text-embedding-3-small",
    )

    # Embed and ingest the chunks/documents into a Chroma db
    # This created "chroma.sqlite3" db file in proj root
    vectorstore = Chroma.from_documents(
        documents=docs_chunks,
        embedding=embedding_function,
        persist_directory=str(Path.cwd()),  # os.getcwd(),
    )

    # Configure the vector store as a retriever object
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )


# Delightful data preparation!
# With your documents split, embedded, and stored, you can now design a prompt
# to combine the retrieved documents and user input.

if __name__ == "__main__":
    # PDF_document_loaders()
    # CSV_document_loaders()
    # HTML_document_loaders()
    # splitting_by_character()
    # recursive_splitting_by_character()
    # splitting_HTML()
    preparing_documents_and_vector_db()
