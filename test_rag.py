from src.rag_chain import create_rag_chain

rag_chain = create_rag_chain()

question = "What is this document about?"

response = rag_chain(question)

print("Answer:")
print(response["result"])