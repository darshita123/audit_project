# retrieval/embeddings.py
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
load_dotenv()

MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
_model = None

def get_embedding_model():
    global _model
    if _model is None:
        print(f"[embeddings] Loading embedding model: {MODEL_NAME}")
        _model = SentenceTransformer(MODEL_NAME)
    return _model

def embed_texts(texts):
    """
    texts: list[str] -> numpy.ndarray (n, d)
    """
    model = get_embedding_model()
    embs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return embs
