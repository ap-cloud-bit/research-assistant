import os
import streamlit as st
from pinecone import Pinecone

# Load secrets
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
INDEX_NAME = st.secrets.get("PINECONE_INDEX_NAME", "research-index")

# Init Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index if missing
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,
        metric="cosine"
    )

index = pc.Index(INDEX_NAME)
