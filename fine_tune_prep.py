import sqlite3, json, re

def redact_pii(text):
    text = re.sub(r'\S+@\S+\.\S+', '[REDACTED_EMAIL]', text)
    text = re.sub(r'\+?\d[\d\-\s]{7,}\d', '[REDACTED_PHONE]', text)
    return text

def export_jsonl(db_path="feedback.db", out_path="fine_tune_input.jsonl"):
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT query, answer FROM feedback WHERE label='yes'").fetchall()
    with open(out_path, "w", encoding="utf-8") as f:
        for q, a in rows:
            q, a = redact_pii(q), redact_pii(a)
            record = {
                "prompt": f"User question:\n{q}\n\nAssistant:",
                "completion": " " + a.strip() + " END"
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Exported {len(rows)} examples to {out_path}")

if __name__ == "__main__":
    export_jsonl()
