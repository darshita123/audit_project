# database/mongo_client.py
from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()

def get_mongo_client():
    mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
    client = MongoClient(mongo_uri)
    return client

def get_database():
    client = get_mongo_client()
    db_name = os.getenv("MONGO_DB_NAME", "audit_ai")
    return client[db_name]

if __name__ == "__main__":
    db = get_database()
    print("Connected to DB:", db.name)
