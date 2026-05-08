import os
import time
import streamlit as st

from src.data_loader import load_pdf_documents
from src.text_splitter import split_documents
from src.retriever import create_vectorstore
from src.rag_chain import create_rag_chain

UPLOAD_FOLDER = "data/uploaded_pdfs"

st.set_page_config(page_title="RAG Chatbot", page_icon="📄")
st.title("Rahul's RAG Chatbot")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

if "messages" not in st.session_state:
    st.session_state["messages"] = []

uploaded_files = st.file_uploader(
    "Upload one or more PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

    st.success("PDF uploaded successfully!")

    if st.button("Process PDFs"):
        with st.spinner("Reading PDFs and creating vector database..."):
            docs = load_pdf_documents(UPLOAD_FOLDER)
            chunks = split_documents(docs)

            persist_directory = f"vectorstore/session_{int(time.time())}"

            create_vectorstore(chunks, persist_directory)

            st.session_state["persist_directory"] = persist_directory
            st.session_state["messages"] = []

        st.success(f"PDFs processed successfully! Created {len(chunks)} chunks.")

st.divider()

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_question = st.chat_input("Ask a question about your documents...")

if user_question:
    if "persist_directory" not in st.session_state:
        st.warning("Please upload and process PDFs first.")
    else:
        st.session_state["messages"].append(
            {"role": "user", "content": user_question}
        )

        with st.chat_message("user"):
            st.write(user_question)

        rag_chain = create_rag_chain(st.session_state["persist_directory"])

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = rag_chain(user_question)
                answer = response["result"]
                st.write(answer)

        st.session_state["messages"].append(
            {"role": "assistant", "content": answer}
        )