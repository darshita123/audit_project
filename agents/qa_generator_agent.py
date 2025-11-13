# agents/qa_generator_agent.py
from crewai import Agent
import pandas as pd
import os, json
from database.logger import log_action
from dotenv import load_dotenv

load_dotenv()

QA_DIR = "datasets/qa_data"
os.makedirs(QA_DIR, exist_ok=True)
COMBINED_QA_PATH = os.path.join(QA_DIR, "combined_qa.json")


def generate_qa_from_labeled_data(json_path):
    """Generate Q&A pairs from labeled bank data and append to a single cumulative file."""
    df = pd.read_json(json_path)
    qa_pairs = []

    for _, row in df.iterrows():
        desc = row.get("DESCRIPTION", "")
        category = row.get("CATEGORY", "")
        if not desc:
            continue
        question = f"What type of transaction is '{desc}'?"
        answer = f"The transaction '{desc}' belongs to the '{category}' category."
        qa_pairs.append({"question": question, "answer": answer})

    # Load existing combined QA file
    if os.path.exists(COMBINED_QA_PATH):
        try:
            with open(COMBINED_QA_PATH, "r") as f:
                existing = json.load(f)
        except Exception:
            existing = []
    else:
        existing = []

    # Append and deduplicate
    existing.extend(qa_pairs)
    unique_data = [dict(t) for t in {tuple(d.items()) for d in existing}]

    with open(COMBINED_QA_PATH, "w") as f:
        json.dump(unique_data, f, indent=2)

    log_action("QAGenerator Agent", "Appended new Q&A data", {"new": len(qa_pairs), "total": len(unique_data)})
    print(f"[qa_generator_agent] ✅ Appended {len(qa_pairs)} new Q&A pairs (total {len(unique_data)})")

    return COMBINED_QA_PATH


qa_generator_agent = Agent(
    role="QA Generator Agent",
    goal="Generate and append Q&A pairs for audit data.",
    backstory="This agent creates incremental Q&A datasets for fine-tuning the local model.",
    llm=None
)