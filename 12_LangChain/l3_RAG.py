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

if __name__ == "__main__":
    PDF_document_loaders()
