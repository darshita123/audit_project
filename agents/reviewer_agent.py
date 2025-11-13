# agents/reviewer_agent.py
from crewai import Agent
import pandas as pd
import os
from database.logger import log_action

def simple_review_check(file_path):
    """
    Review labeled data for consistency and accuracy.
    Returns a structured JSON: {"accuracy": float, "comments": str}
    """
    try:
        df = pd.read_csv(file_path)

        # Basic rule-based quality metrics
        total = len(df)
        if total == 0:
            return {"accuracy": 0.0, "comments": "Empty dataset."}

        # Check for missing categories
        missing = df["CATEGORY"].isna().sum() if "CATEGORY" in df.columns else total
        missing_ratio = missing / total

        # Calculate an approximate accuracy
        accuracy = round(max(0.0, 1.0 - missing_ratio - 0.05), 2)
        comments = []

        if missing_ratio > 0.1:
            comments.append("Too many unlabeled transactions.")
        if any(df["CATEGORY"].str.lower().str.contains("other", na=False)):
            comments.append("Contains generic 'Other' categories — may need tuning.")
        if accuracy >= 0.9:
            comments.append("Excellent labeling quality.")
        elif accuracy >= 0.8:
            comments.append("Acceptable, but can be improved.")
        else:
            comments.append("Poor labeling quality — fine-tuning recommended.")

        review = {"accuracy": accuracy, "comments": " ".join(comments)}
        log_action("Reviewer Agent", "Review completed", {"accuracy": accuracy, "missing": missing})
        print(f"[Reviewer Agent] ✅ Review complete — Accuracy: {accuracy}")

        return review

    except Exception as e:
        print(f"[Reviewer Agent] ❌ Error: {e}")
        return {"accuracy": 0.0, "comments": f"Error during review: {e}"}


reviewer_agent = Agent(
    role="Reviewer Agent",
    goal="Review labeled data and provide feedback on model performance.",
    backstory="The Reviewer Agent checks data quality and decides if fine-tuning is needed.",
    llm=None
)