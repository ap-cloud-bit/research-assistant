import os, uuid
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from pinecone import Pinecone
import openai

# Load API keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-west1-gcp")
INDEX_NAME = "research-papers"

openai.api_key = OPENAI_API_KEY
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# If index doesn’t exist yet:
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(name=INDEX_NAME, dimension=1536, metric="cosine")

index = pinecone.Index(INDEX_NAME)
emb = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

def ingest_pdf(path, meta={}):
    """Ingest a PDF into Pinecone index."""
    loader = PyPDFLoader(path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    to_upsert = []
    for doc in chunks:
        vector = emb.embed_documents([doc.page_content])[0]
        metadata = {
            "source": meta.get("source", path),
            "title": meta.get("title", "Unknown Title"),
            "page": doc.metadata.get("page", None),
            "text": doc.page_content,
            "chunk_id": str(uuid.uuid4())
        }
        to_upsert.append((metadata["chunk_id"], vector, metadata))

    index.upsert(vectors=to_upsert)
    print(f"Ingested {len(to_upsert)} chunks from {path}")

def retrieve_and_answer(query, top_k=5):
    """Retrieve sources and generate citation-first answer."""
    q_emb = emb.embed_query(query)
    res = index.query(queries=[q_emb], top_k=top_k, include_metadata=True)
    hits = res["matches"]

    sources, context_text = [], ""
    for i, h in enumerate(hits):
        m = h["metadata"]
        cite = f"[{i+1}] {m['title']} (page {m.get('page')})"
        sources.append({"cite": cite, "chunk_id": m["chunk_id"]})
        context_text += f"\n\n### Source {i+1}\n{m['text']}"

    prompt = f"""You are a research assistant.
Return a citation-first answer.

SOURCES:
{chr(10).join([s['cite'] for s in sources])}

USER QUESTION:
{query}

CONTEXT:
{context_text}

Now answer with a concise summary grounded in the context, then list CITED SOURCES by number.
"""

    resp = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role":"system","content":"You are a helpful research assistant."},
                  {"role":"user","content":prompt}],
        temperature=0.2
    )

    return {"sources": sources, "answer": resp["choices"][0]["message"]["content"]}
