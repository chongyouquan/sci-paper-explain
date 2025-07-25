# Scientific Paper Explainer - Small Example (RAG using LangChain + Vector Search using FAISS)

import os
import faiss
import openai
from typing import List
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyMuPDFLoader
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# Setup OpenAI key
openai.api_key = os.environ.get("OPENAI_API_KEY")

# --- 1. Load and Split PDF ---
def load_and_split_pdf(path: str):
    loader = PyMuPDFLoader(path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    return chunks

# --- 2. Create Vector Store ---
def create_vector_store(chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

# --- 3. Create RAG QA Chain ---
def create_qa_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True
    )
    return qa_chain

# --- 4. Ask a Question ---
def ask_question(qa_chain, question: str):
    result = qa_chain({"query": question})
    print("Answer:\n", result["result"])
    print("\nSources:")
    for doc in result["source_documents"]:
        print("-", doc.metadata.get("source", "Unknown"), "...", doc.page_content[:150], "...")

# --- Example Usage ---
if __name__ == "__main__":
    pdf_path = "example_paper.pdf"  # <-- Replace with your local PDF file
    print("Loading and splitting PDF...")
    chunks = load_and_split_pdf(pdf_path)

    print("Creating vector store...")
    vs = create_vector_store(chunks)

    print("Creating QA chain...")
    qa = create_qa_chain(vs)

    print("\nAsk your question:")
    q = input(">> ")
    ask_question(qa, q)
