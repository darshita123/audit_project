# agents/labeling_agent.py
from crewai import Agent
import pandas as pd
import os, json
from database.logger import log_action
from dotenv import load_dotenv
load_dotenv()

from core.model_loader import load_phi3_model, generate_with_model

MODEL_BACKEND = os.getenv("MODEL_BACKEND", "transformers")
MODEL_NAME = os.getenv("MODEL_NAME", "microsoft/phi-3-mini-4k-instruct")

# load model once
if MODEL_BACKEND == "transformers":
    model, tokenizer = load_phi3_model()

def detect_category(desc: str):
    if not isinstance(desc, str): return None
    d = desc.lower()
    if "atm" in d or "cash" in d: return "Cash Withdrawal"
    if "salary" in d and ("credit" in d or "salary" in d): return "Income"
    if "amazon" in d or "flipkart" in d or "myntra" in d: return "Shopping"
    if "interest" in d: return "Interest"
    return None

def label_with_llm(description: str):
    prompt = (
        "You are a helpful assistant. Classify the following bank transaction description into one of:\n"
        "[Income, Expense, Shopping, Interest, Transfer, Cash Withdrawal, Other(whenever you are specifying other category you should also specify the sub category also in bracket)]\n\n"
        f"Transaction: \"{description}\"\n\n"
        "Answer with the single category name."
    )
    try:
        return generate_with_model(model, tokenizer, prompt, max_new_tokens=32).split("\n")[0].strip()
    except Exception as e:
        print("[label_with_llm] LLM error:", e)
        return "Other"

def label_bank_statement(file_path):
    df = pd.read_csv(file_path)
    if "DESCRIPTION" not in df.columns:
        raise ValueError("CSV must contain DESCRIPTION column.")
    categories = []
    for desc in df["DESCRIPTION"].astype(str):
        rule = detect_category(desc)
        print(rule)
        if rule:
            print("i am an rule based agent")
            categories.append(rule)
        else:
            cat = label_with_llm(desc)
            categories.append(cat if cat else "Other")
    df["CATEGORY"] = categories

    os.makedirs("datasets/labeled_data", exist_ok=True)
    out_csv = os.path.join("datasets/labeled_data", os.path.basename(file_path).replace(".csv","_labeled.csv"))
    df.to_csv(out_csv, index=False)
    json_path = out_csv.replace(".csv", ".json")
    df.to_json(json_path, orient="records", indent=2)

    log_action("Labeling Agent", "Labeled document", {"file": file_path, "output_csv": out_csv, "rows": len(df), "backend": MODEL_BACKEND})
    return out_csv

labeling_agent = Agent(
    role="Labeling Agent",
    goal="Label and categorize fields in bank statements using rule + local LLM.",
    backstory="Hybrid labeling agent.",
    llm=None
)
