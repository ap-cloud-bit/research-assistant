# utils/db_utils.py
import os
from typing import List, Optional
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from pinecone import Pinecone, ServerlessSpec
from langchain.docstore.document import Document

def init_pinecone() -> tuple:
    api_key = os.getenv("PINECONE_API_KEY")
    env = os.getenv("PINECONE_ENVIRONMENT") or os.getenv("PINECONE_ENV")
    if not api_key:
        raise EnvironmentError("PINECONE_API_KEY not set in environment")

    pc = Pinecone(api_key=api_key)

    index_name = os.getenv("PINECONE_INDEX_NAME", "research-index")
    if index_name not in [i["name"] for i in pc.list_indexes()]:
        dim = 384  # HuggingFace MiniLM embeddings
        pc.create_index(
            name=index_name,
            dimension=dim,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=env or "us-west-2")
        )
    return pc, index_name

def build_or_load_vectorstore(
    docs: Optional[List[Document]] = None,
    namespace: Optional[str] = None
):
    pc, index_name = init_pinecone()

    # ✅ HuggingFace embeddings
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if docs:
        # Store documents into Pinecone
        vect = LangchainPinecone.from_documents(
            documents=docs,
            embedding=emb,
            index_name=index_name,
            namespace=namespace
        )
    else:
        # Load existing Pinecone index
        vect = LangchainPinecone.from_existing_index(
            index_name=index_name,
            embedding=emb,
            namespace=namespace
        )
    return vect
