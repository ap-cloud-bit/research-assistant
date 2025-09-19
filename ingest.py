from pinecone import Pinecone
import streamlit as st

PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
INDEX_NAME = st.secrets.get("PINECONE_INDEX_NAME", "research-index")

pc = Pinecone(api_key=PINECONE_API_KEY)

# Check if index already exists
existing_indexes = pc.list_indexes().names()

if INDEX_NAME not in existing_indexes:
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,
        metric="cosine"
    )

# Connect to index
index = pc.Index(INDEX_NAME)
