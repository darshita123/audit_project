from datetime import datetime
from database.mongo_client import get_database
import numpy as np

def convert_numpy(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    return obj

def log_action(agent_name, action, details):
    db = get_database()
    
    # Convert any numpy types in details
    safe_details = convert_numpy(details)
    
    log_entry = {
        "agent": agent_name,
        "action": action,
        "details": safe_details,
        "timestamp": datetime.utcnow()
    }
    
    db.logs.insert_one(log_entry)
    print(f"[LOG] {agent_name} → {action}")