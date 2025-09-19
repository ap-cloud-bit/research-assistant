import streamlit as st
from openai import OpenAI
from pinecone import Pinecone

# Load secrets
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
INDEX_NAME = st.secrets["PINECONE_INDEX_NAME"]

# Initialize clients
client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

# Connect to Pinecone index
index = pc.Index(INDEX_NAME)

def retrieve_and_answer(query: str) -> str:
    """
    Retrieve relevant docs from Pinecone and generate an answer with OpenAI.
    """
    # Step 1: Embed the query
    embed = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    query_vector = embed.data[0].embedding

    # Step 2: Search Pinecone
    results = index.query(vector=query_vector, top_k=3, include_metadata=True)

    # Step 3: Build context from retrieved docs
    context = "\n".join([m["metadata"].get("text", "") for m in results["matches"]])

    # Step 4: Ask OpenAI with context
    prompt = f"Answer the question based on the context:\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content
