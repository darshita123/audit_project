# agents/executor_agent.py
from crewai import Agent
from agents.labeling_agent import label_bank_statement
from database.logger import log_action
from retrieval.rag import index_labeled_file, retrieve_and_answer
from retrieval.generators import generator_fn
import os

def execute_audit(file_path):
    labeled_csv = label_bank_statement(file_path)
    labeled_json = labeled_csv.replace(".csv", ".json")
    index_info = index_labeled_file(labeled_json)
    log_action("Executor Agent", "Executed labeling and indexing", {"labeled_csv": labeled_csv, "index": index_info})
    return labeled_csv

def answer_query(query_text, top_k=None):
    top_k = top_k or int(os.getenv("RAG_TOP_K", 5))
    answer, results = retrieve_and_answer(query_text, generator_fn, top_k=top_k)
    return answer, results

executor_agent = Agent(
    role="Executor Agent",
    goal="Parse, label, and summarize financial data.",
    backstory="Executor that runs labeling and indexing tasks.",
    llm=None
)
