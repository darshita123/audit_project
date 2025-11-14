---
# Audit Intelligence System

This project is a **Review Logging and labeling Statements** built with Python, CrewAI, MongoDB, and Streamlit.  
It allows agents to perform reviews, perform plan and execute agents and log results into a MongoDB database, and track performance metrics like accuracy and missing counts. It's an LLM based application that works on
"google/flan-t5-base" it's an SLM and this provides fine tuning functionality of this model based on accuracy.

---

## Table of Contents

- [Features](#features)  
- [Requirements](#requirements)  
- [Setup](#setup)  
- [Running the Project](#running-the-project)  
  - [Run via Python Script](#run-via-python-script)  
  - [Run via Streamlit](#run-via-streamlit) 


---

## Requirements

- Python 3.10+  
- MongoDB (local install or Docker container)

Install dependencies via:

bash
pip install -r requirements.txt


---

## Setup

1. **Clone the repository**:

bash
[(https://github.com/darshita123/audit_project.git)](https://github.com/darshita123/audit_project.git)


2. **Create a virtual environment** (recommended):

bash
python -m venv .venv


Activate it:

* Windows:

bash
.venv\Scripts\activate


* Linux / Mac:

bash
source .venv/bin/activate


3. **Install dependencies**:

bash
pip install -r requirements.txt


4. **Run MongoDB using Docker** (optional, recommended):

* Pull the latest MongoDB image:

bash
docker pull mongo:latest


* Run a MongoDB container:

bash
docker run -d -p 27017:27017 --name rag_mongo mongo:latest


* Verify it’s running:

bash
docker ps


* (Optional) For persistent data storage:

bash
docker run -d -p 27017:27017 --name rag_mongo -v mongo_data:/data/db mongo:latest


* To stop and remove the container:

bash
docker stop rag_mongo
docker rm rag_mongo


5. **Configure MongoDB connection**:

Edit `database/mongo_client.py` to match your MongoDB URI:

python
from pymongo import MongoClient

def get_database():
    client = MongoClient("mongodb://localhost:27017/")  # Docker or local MongoDB
    return client["your_database_name"]


---

## Running the Project

### Run via Python Script

You can run individual Python scripts that use the logging system:

bash
python crew_setup.py datasets\"your_dataset_path"


* Logs will be inserted into the `logs` collection in MongoDB.
* Works with both native Python types and `numpy` types.

---

### Run via Streamlit

The Streamlit interface allows you to log review actions via a web UI:

bash
streamlit run app.py


* Open the URL provided in the terminal (default: `http://localhost:8501`)
* Perform actions and see logs printed in the console and stored in MongoDB

---

## Contributing

1. Fork the repository
2. Create a new branch (`git checkout -b feature-name`)
3. Make your changes
4. Commit (`git commit -m "Add feature"`)
5. Push (`git push origin feature-name`)
6. Open a Pull Request

---


```
