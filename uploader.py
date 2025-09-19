import streamlit as st
from openai import OpenAI
from pinecone import Pinecone
from pypdf import PdfReader
import os

# Load secrets
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
INDEX_NAME = st.secrets["PINECONE_INDEX_NAME"]

# Initialize clients
client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

st.title("📄 Document Uploader & Ingestor")

uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

if uploaded_file is not None:
    if uploaded_file.type == "application/pdf":
        reader = PdfReader(uploaded_file)
        text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
    else:
        text = uploaded_file.read().decode("utf-8")

    st.success("✅ File loaded. Splitting into chunks...")

    # Split into chunks (so embeddings fit in context)
    chunk_size = 500
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    # Embed and upload
    vectors = []
    for i, chunk in enumerate(chunks):
        embed = client.embeddings.create(
            model="text-embedding-3-small",
            input=chunk
        )
        vector = {
            "id": f"{uploaded_file.name}-{i}",
            "values": embed.data[0].embedding,
            "metadata": {"text": chunk, "source": uploaded_file.name}
        }
        vectors.append(vector)

    # Upsert to Pinecone
    index.upsert(vectors=vectors)
    st.success(f"✅ Uploaded {len(vectors)} chunks from {uploaded_file.name} to Pinecone.")
