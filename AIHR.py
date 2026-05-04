pip install langchain langchain-community chromadb sentence-transformers transformers accelerate

import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import pipeline

# -------------------------------
# 1. LOAD DOCUMENT
# -------------------------------
loader = TextLoader("leave_policy.txt")  # save your doc as this
documents = loader.load()

# -------------------------------
# 2. SPLIT INTO CHUNKS
# -------------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

docs = text_splitter.split_documents(documents)

# -------------------------------
# 3. CREATE EMBEDDINGS
# -------------------------------
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -------------------------------
# 4. VECTOR STORE (CHROMA)
# -------------------------------
vectorstore = Chroma.from_documents(
    docs,
    embedding_model,
    persist_directory="./chroma_db"
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# -------------------------------
# 5. LOAD OPEN-SOURCE LLM
# -------------------------------
hf_pipeline = pipeline(
    "text-generation",
    model="tiiuae/falcon-7b-instruct",  # can change if heavy
    max_new_tokens=300,
    temperature=0.2
)

llm = HuggingFacePipeline(pipeline=hf_pipeline)

# -------------------------------
# 6. RULE-BASED LAYER (IMPORTANT)
# -------------------------------
def apply_rules(query, response):
    query_lower = query.lower()

    # Example rule 1: casual leave limit
    if "casual leave" in query_lower and "more than 3 days" in query_lower:
        return "Casual Leave cannot exceed 3 consecutive days as per policy."

    # Example rule 2: carry forward limit
    if "carry forward" in query_lower:
        return "Earned Leave can only be carried forward up to 45 days."

    # Example rule 3: notice period restriction
    if "notice period" in query_lower:
        return "Leave during notice period is discouraged and subject to approval."

    return response

# -------------------------------
# 7. RAG QA CHAIN
# -------------------------------
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# -------------------------------
# 8. QUERY FUNCTION
# -------------------------------
def ask_question(query):
    result = qa_chain(query)
    
    raw_answer = result["result"]
    
    # Apply rules AFTER LLM
    final_answer = apply_rules(query, raw_answer)

    print("\n Question:", query)
    print("\n Answer:", final_answer)
    print("\n Sources:")
    for doc in result["source_documents"]:
        print("-", doc.metadata)

# -------------------------------
# 9. TEST
# -------------------------------
if __name__ == "__main__":
    while True:
        q = input("\nAsk HR Assistant: ")
        if q.lower() == "exit":
            break
        ask_question(q)

  model="google/flan-t5-base"
