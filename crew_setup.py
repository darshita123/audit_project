# crew_setup.py
print("Starting crew_setup.py...")
from agents.planner_agent import plan_task
from agents.executor_agent import execute_audit, answer_query
from agents.qa_generator_agent import generate_qa_from_labeled_data
from database.logger import log_action
import os

def run_audit_query(file_path):
    query = f"Analyze and label bank statement: {file_path}"
    print(query)
    plan = plan_task(query)
    print(plan)
    log_action("Crew", "Received plan", {"steps": plan})
    labeled_csv = execute_audit(file_path)
    labeled_json = labeled_csv.replace(".csv", ".json")
    qa_file = generate_qa_from_labeled_data(labeled_json)
    log_action("Crew", "Completed audit flow", {"labeled_csv": labeled_csv, "qa_file": qa_file})
    return {"labeled_file": labeled_csv, "qa_file": qa_file}

if __name__ == "__main__":
    import sys
    fp = sys.argv[1] if len(sys.argv) > 1 else "datasets/bank_statement.csv"
    print(f"\n🚀 Starting audit analysis for: {fp}\n")
    try:
        result = run_audit_query(fp)
        print("\n✅ Audit completed successfully!", result)
    except Exception as e:
        import traceback
        print("\n❌ ERROR DURING RUN:")
        traceback.print_exc()
