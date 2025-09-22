# ingest.py
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub
from utils.db_utils import build_or_load_vectorstore

# Load PDFs from the data folder
def load_papers(data_dir="data/papers"):
    docs = []
    for fname in os.listdir(data_dir):
        if fname.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(data_dir, fname))
            docs.extend(loader.load())
    return docs

# Build index or load existing one
def ingest_papers():
    docs = load_papers()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)
    vectorstore = build_or_load_vectorstore(split_docs)
    return vectorstore

# Retrieve and answer
def retrieve_and_answer(query: str, top_k: int = 5):
    vectorstore = build_or_load_vectorstore([])  # load existing index
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

    # ✅ Use a free HuggingFace model for QA
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-large",   # you can swap with other models
        model_kwargs={"temperature": 0, "max_length": 512}
    )

    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
    result = qa.invoke({"query": query})

    # Format answer + citations
    answer = result["result"]
    sources = [
        {"cite": d.metadata.get("source", "Unknown"), "content": d.page_content[:200]}
        for d in result["source_documents"]
    ]

    return {"answer": answer, "sources": sources}
