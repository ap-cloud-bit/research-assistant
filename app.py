import streamlit as st
import sqlite3, uuid, time, json, os
from ingest import retrieve_and_answer, ingest_papers

DB_PATH = "feedback.db"

# Initialize DB
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
conn.execute("""
CREATE TABLE IF NOT EXISTS feedback (
    id TEXT PRIMARY KEY,
    query TEXT,
    answer TEXT,
    sources TEXT,
    label TEXT,
    comment TEXT,
    ts INT
)
""")
conn.commit()

st.set_page_config(page_title="Research Assistant", layout="wide")
st.title("📚 Research Assistant (Citation-first)")

# ============================================================
# 📂 Sidebar uploader & ingestion
# ============================================================
st.sidebar.header("📂 Upload & Ingest Papers")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDFs", type=["pdf"], accept_multiple_files=True
)

if uploaded_files:
    os.makedirs("data/papers", exist_ok=True)
    for f in uploaded_files:
        file_path = os.path.join("data/papers", f.name)
        with open(file_path, "wb") as out_file:
            out_file.write(f.getbuffer())
        st.sidebar.success(f"✅ Saved {f.name}")

    if st.sidebar.button("Ingest uploaded papers"):
        with st.spinner("Ingesting PDFs into Pinecone..."):
            ingest_papers()
        st.sidebar.success("✅ Papers ingested successfully!")

# ============================================================
# 🔎 Main Q&A
# ============================================================
query = st.text_area("Ask a research question", height=120)

if st.button("🔎 Get Answer"):
    result = retrieve_and_answer(query, top_k=5)

    st.markdown("### Sources")
    if result["sources"]:
        for s in result["sources"]:
            st.markdown(f"- {s['cite']}")
    else:
        st.warning("⚠️ No sources found. Did you ingest any PDFs?")

    st.markdown("---")
    st.markdown("### Answer")
    st.write(result["answer"])

    st.markdown("### Feedback")
    col1, col2 = st.columns(2)
    if col1.button("👍 Helpful"):
        record = (
            str(uuid.uuid4()), query, result["answer"],
            json.dumps(result["sources"]), "yes", "", int(time.time())
        )
        conn.execute("INSERT INTO feedback VALUES (?,?,?,?,?,?,?)", record)
        conn.commit()
        st.success("✅ Saved as positive example.")

    if col2.button("👎 Not Helpful"):
        comment = st.text_area("Optional: Tell us what went wrong")
        if st.button("Submit negative feedback"):
            record = (
                str(uuid.uuid4()), query, result["answer"],
                json.dumps(result["sources"]), "no", comment, int(time.time())
            )
            conn.execute("INSERT INTO feedback VALUES (?,?,?,?,?,?,?)", record)
            conn.commit()
            st.warning("⚠️ Saved as negative feedback.")
