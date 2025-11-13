# agents/planner_agent.py
from crewai import Agent
from database.logger import log_action

def plan_task(user_query):
    steps = [
        "Parse document",
        "Extract transactions",
        "Auto-label transactions",
        "Index labeled data into RAG",
        "Generate Q&A pairs",
        "Optionally fine-tune the model"
    ]
    log_action("Planner Agent", "Plan created", {"query": user_query, "steps": steps})
    return steps

planner_agent = Agent(
    role="Planner Agent",
    goal="Break user queries into sequential audit subtasks.",
    backstory="Planner that decomposes audit workflows.",
    llm=None
)
