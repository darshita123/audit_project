# database/logger.py
from datetime import datetime
from database.mongo_client import get_database
import traceback

def log_action(agent_name, action, details):
    try:
        db = get_database()
        log_entry = {
            "agent": agent_name,
            "action": action,
            "details": details,
            "timestamp": datetime.utcnow()
        }
        db.logs.insert_one(log_entry)
        print(f"[LOG] {agent_name} → {action}")
    except Exception as e:
        print("[LOG ERROR]", e)
        traceback.print_exc()
