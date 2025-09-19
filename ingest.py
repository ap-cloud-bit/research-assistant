from pinecone import Pinecone
import streamlit as st

# Load API keys & index name
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
INDEX_NAME = st.secrets.get("PINECONE_INDEX_NAME", "research-index")

# Init Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Connect to your existing index
index = pc.Index(INDEX_NAME)
