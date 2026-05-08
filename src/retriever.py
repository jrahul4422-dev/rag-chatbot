import os
from langchain_chroma import Chroma
from src.embeddings import get_embeddings


def create_vectorstore(chunks, persist_directory):
    os.makedirs(persist_directory, exist_ok=True)

    embeddings = get_embeddings()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )

    return vectorstore


def get_retriever(persist_directory):
    embeddings = get_embeddings()

    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 5}
    )

    return retriever