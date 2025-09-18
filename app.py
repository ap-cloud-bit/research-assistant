import streamlit as st
import sqlite3, uuid, time, json, os
from ingest import retrieve_and_answer

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

query = st.text_area("Ask a research question", height=120)

if st.button("🔎 Get Answer"):
    result = retrieve_and_answer(query, top_k=5)
    st.markdown("### Sources")
    for s in result["sources"]:
        st.markdown(f"- {s['cite']}")
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
