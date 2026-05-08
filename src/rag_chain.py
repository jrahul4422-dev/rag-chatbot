from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from src.retriever import get_retriever

load_dotenv()


def create_rag_chain(persist_directory):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    retriever = get_retriever(persist_directory)

    def rag_answer(question):
        docs = retriever.invoke(question)

        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"""
You are a helpful AI assistant.
Answer the question using only the context below.

Context:
{context}

Question:
{question}
"""

        response = llm.invoke(prompt)

        return {
            "result": response.content,
            "source_documents": docs
        }

    return rag_answer