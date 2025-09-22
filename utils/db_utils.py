# utils/db_utils.py
"""
Pinecone helper for building/loading a LangChain vectorstore.

Usage:
- To ingest docs: build_or_load_vectorstore(docs=your_docs_list)
- To query existing index: build_or_load_vectorstore(docs=None)

Requires these environment variables:
- PINECONE_API_KEY
- PINECONE_ENVIRONMENT (or PINECONE_ENV)
- PINECONE_INDEX_NAME  (optional; default "research-index")
- PINECONE_INDEX_DIM   (optional; default "1536")
- OPENAI_API_KEY       (for OpenAIEmbeddings)
"""
import os
from typing import List, Optional
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.docstore.document import Document


def init_pinecone() -> str:
    api_key = os.getenv("PINECONE_API_KEY")
    env = os.getenv("PINECONE_ENVIRONMENT") or os.getenv("PINECONE_ENV")
    if not api_key:
        raise EnvironmentError("PINECONE_API_KEY not set in environment")
    pinecone.init(api_key=api_key, environment=env)
    index_name = os.getenv("PINECONE_INDEX_NAME", "research-index")
    if index_name not in pinecone.list_indexes():
        dim = int(os.getenv("PINECONE_INDEX_DIM", "1536"))
        pinecone.create_index(name=index_name, dimension=dim, metric="cosine")
    return index_name


def build_or_load_vectorstore(docs: Optional[List[Document]] = None, namespace: Optional[str] = None):
    """
    If `docs` is provided, upsert them into Pinecone (creates index if missing).
    If `docs` is None, return a LangChain Pinecone wrapper for the existing index.
    """
    index_name = init_pinecone()
    emb = OpenAIEmbeddings()  # uses OPENAI_API_KEY env var

    # If we have docs, create/upsert them
    if docs:
        vect = Pinecone.from_documents(documents=docs, embedding=emb, index_name=index_name, namespace=namespace)
    else:
        # Load wrapper for existing index
        vect = Pinecone.from_existing_index(index_name=index_name, embedding=emb, namespace=namespace)
    return vect
