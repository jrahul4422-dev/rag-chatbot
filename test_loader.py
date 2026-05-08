from src.data_loader import load_pdf_documents
from src.text_splitter import split_documents
from src.retriever import create_vectorstore

docs = load_pdf_documents("data/uploaded_pdfs")
chunks = split_documents(docs)

vectorstore = create_vectorstore(chunks)

print(f"Loaded {len(docs)} pages")
print(f"Created {len(chunks)} chunks")
print("Vector database created successfully")