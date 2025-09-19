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

def retrieve_and_answer(query: str, top_k: int = 3) -> dict:
    """Retrieve docs from Pinecone and generate an answer with citations."""
    try:
        # Step 1: Embed the query
        embed = client.embeddings.create(
            model="text-embedding-3-small",
            input=query
        )
        query_vector = embed.data[0].embedding

        # Step 2: Search Pinecone
        results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)

        # Step 3: Collect sources
        sources = []
        context_parts = []
        for match in results.get("matches", []):
            text = match["metadata"].get("text", "")
            cite = match["metadata"].get("source", "Unknown source")
            context_parts.append(text)
            sources.append({"text": text, "cite": cite})

        context = "\n".join(context_parts)

        # Step 4: Ask OpenAI
        prompt = f"Answer the question based on the context:\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer with citations if possible:"
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        answer = response.choices[0].message.content

        return {
            "answer": answer,
            "sources": sources
        }

    except Exception as e:
        return {
            "answer": f"⚠️ Error: {str(e)}",
            "sources": []
        }
