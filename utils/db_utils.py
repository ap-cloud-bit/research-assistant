# utils/db_utils.py
import os
from typing import List, Optional
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone as LangchainPinecone
import pinecone  # ✅ v2 client
from langchain.docstore.document import Document

def init_pinecone() -> str:
    api_key = os.getenv("PINECONE_API_KEY")
    env = os.getenv("PINECONE_ENVIRONMENT") or os.getenv("PINECONE_ENV")
    if not api_key:
        raise EnvironmentError("PINECONE_API_KEY not set in environment")

    pinecone.init(api_key=api_key, environment=env)

    index_name = os.getenv("PINECONE_INDEX_NAME", "research-index")
    if index_name not in pinecone.list_indexes():
        dim = 384  # HuggingFace MiniLM embeddings dimension
        pinecone.create_index(
            name=index_name,
            dimension=dim,
            metric="cosine"
        )
    return index_name

def build_or_load_vectorstore(
    docs: Optional[List[Document]] = None,
    namespace: Optional[str] = None
):
    index_name = init_pinecone()

    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if docs:
        vect = LangchainPinecone.from_documents(
            documents=docs,
            embedding=emb,
            index_name=index_name,
            namespace=namespace
        )
    else:
        vect = LangchainPinecone.from_existing_index(
            index_name=index_name,
            embedding=emb,
            namespace=namespace
        )
    return vect
