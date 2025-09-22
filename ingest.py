import os
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from utils.db_utils import build_or_load_vectorstore


def retrieve_and_answer(query, top_k=5):
    """
    Retrieve context from Pinecone and generate an answer using Groq LLM.
    Returns dict with 'answer' and 'sources'.
    """

    # Build or load Pinecone vectorstore
    vectorstore = build_or_load_vectorstore([])  # docs=[] means "just load existing index"
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

    # Initialize Groq LLM
    llm = ChatGroq(
        groq_api_key=os.getenv("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"),
        model_name="llama3-70b-8192",   # alt: "llama3-8b-8192"
        temperature=0
    )

    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True
    )

    # Run query
    result = qa_chain(query)
    answer = result["result"]
    sources = [
        {"cite": doc.metadata.get("source", "unknown")}
        for doc in result["source_documents"]
    ]

    return {"answer": answer, "sources": sources}
