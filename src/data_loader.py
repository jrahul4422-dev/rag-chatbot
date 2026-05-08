import os
from langchain_community.document_loaders import PyPDFLoader


def load_pdf_documents(folder_path: str = "data/uploaded_pdfs"):
    """
    Loads all PDF files from the given folder and returns LangChain documents.
    """

    documents = []

    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".pdf"):
            file_path = os.path.join(folder_path, file_name)

            loader = PyPDFLoader(file_path)
            pdf_docs = loader.load()

            documents.extend(pdf_docs)

    if not documents:
        raise ValueError("No PDF documents found. Please add PDF files to data/uploaded_pdfs")

    return documents