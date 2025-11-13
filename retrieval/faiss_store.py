# retrieval/faiss_store.py
import faiss
import numpy as np
import os, json
from database.mongo_client import get_database
from dotenv import load_dotenv
from datetime import datetime
from bson import ObjectId
load_dotenv()

INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "outputs/faiss/faiss_index.idx")
META_PATH = os.getenv("FAISS_META_PATH", "outputs/faiss/faiss_meta.json")
DB = get_database()

def create_index(d):
    # Use inner product + normalized vectors for cosine similarity
    return faiss.IndexFlatIP(d)

def save_index(index):
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    faiss.write_index(index, INDEX_PATH)

def load_index():
    if os.path.exists(INDEX_PATH):
        return faiss.read_index(INDEX_PATH)
    return None

def persist_meta(meta):
    os.makedirs("outputs/meta", exist_ok=True)

    # ✅ recursively convert ObjectIds to strings
    def convert(o):
        if isinstance(o, ObjectId):
            return str(o)
        if isinstance(o, list):
            return [convert(x) for x in o]
        if isinstance(o, dict):
            return {k: convert(v) for k, v in o.items()}
        return o

    meta_serializable = convert(meta)

    with open("outputs/meta/meta.json", "w") as f:
        json.dump(meta_serializable,f,indent=2)

def load_meta():
    if os.path.exists(META_PATH):
        with open(META_PATH, "r") as f:
            return json.load(f)
    return []

def index_documents(texts, metadatas, embeddings):
    """
    texts: list[str]
    metadatas: list[dict]
    embeddings: np.ndarray shape (n,dim)
    """
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    faiss.normalize_L2(embeddings)
    index = load_index()
    dim = embeddings.shape[1]
    if index is None:
        index = create_index(dim)
    start_id = index.ntotal
    index.add(embeddings)
    save_index(index)

    meta = load_meta()
    for i, m in enumerate(metadatas):
        meta_entry = {
            "vector_id": start_id + i,
            "metadata": m,
            "indexed_at": datetime.utcnow().isoformat()
        }
        meta.append(meta_entry)
        try:
            DB.rag_metadata.insert_one(meta_entry)
        except Exception:
            pass
    persist_meta(meta)
    return start_id, start_id + len(metadatas) - 1

def retrieve(query_embedding, top_k=5):
    index = load_index()
    if index is None or index.ntotal == 0:
        return []
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)
    faiss.normalize_L2(query_embedding)
    D, I = index.search(query_embedding, top_k)
    ids = I[0].tolist()
    meta = load_meta()
    results = []
    for vid in ids:
        for item in meta:
            if item["vector_id"] == vid:
                results.append(item)
                break
    return results
