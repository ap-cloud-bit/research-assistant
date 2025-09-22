# ingest.py
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub
from utils.db_utils import build_or_load_vectorstore

def load_papers(data_dir="data/papers"):
    """Load all PDFs from the data/papers directory."""
    docs = []
    if not os.path.exists(data_dir):
        print(f"⚠️ Data directory not found: {data_dir}")
        return docs

    for fname in os.listdir(data_dir):
        if fname.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(data_dir, fname))
            docs.extend(loader.load())
    return docs

def ingest_papers():
    """Read PDFs, split into chunks, and upload to Pinecone."""
    docs = load_papers()
    if not docs:
        print("⚠️ No PDFs found in data/papers/. Nothing ingested.")
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)

    print(f"📥 Ingesting {len(split_docs)} chunks into Pinecone...")
    vectorstore = build_or_load_vectorstore(split_docs)
    return vectorstore

def retrieve_and_answer(query: str, top_k: int = 5):
    """Retrieve context from Pinecone and answer using HuggingFace LLM."""
    try:
        vectorstore = build_or_load_vectorstore([])  # Load existing index
    except Exception as e:
        return {
            "answer": "⚠️ Error: Could not connect to Pinecone index. Did you ingest any PDFs?",
            "sources": []
        }

    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

    # ✅ Use HuggingFace Hub for free models
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-large",  # you can change to llama-2, mistral, etc.
        model_kwargs={"temperature": 0, "max_length": 512}
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    result = qa.invoke({"query": query})
    answer = result.get("result", "⚠️ No answer generated.")
    sources = [
        {"cite": d.metadata.get("source", "Unknown"), "content": d.page_content[:200]}
        for d in result.get("source_documents", [])
    ]

    return {"answer": answer, "sources": sources}
