# retrieval/rag.py
import json, os
from retrieval.embeddings import embed_texts
from retrieval.faiss_store import index_documents, retrieve
from database.logger import log_action

def chunk_text_rows_from_labeled_json(labeled_json_path):
    with open(labeled_json_path, "r") as f:
        data = json.load(f)
    texts, meta = [], []
    for i, txn in enumerate(data):
        text = f"{txn.get('DATE','')} {txn.get('DESCRIPTION','')} DEBIT:{txn.get('DEBIT','')} CREDIT:{txn.get('CREDIT','')} BALANCE:{txn.get('BALANCE','')} CATEGORY:{txn.get('CATEGORY','')}"
        texts.append(text)
        meta.append({
            "doc_id": os.path.basename(labeled_json_path),
            "row_index": i,
            "description": txn.get("DESCRIPTION",""),
            "category": txn.get("CATEGORY","")
        })
    return texts, meta

def index_labeled_file(labeled_json_path):
    texts, metadatas = chunk_text_rows_from_labeled_json(labeled_json_path)
    if not texts:
        return None
    embeddings = embed_texts(texts)
    start, end = index_documents(texts, metadatas, embeddings)
    log_action("RAG", "Indexed file", {"file": labeled_json_path, "rows_indexed": len(texts)})
    return {"start": start, "end": end, "count": len(texts)}

def retrieve_and_answer(query_text, generator_fn, top_k=5):
    query_emb = embed_texts([query_text])
    results = retrieve(query_emb, top_k)
    context = "\n\n".join([f"{r['metadata']['description']} (category: {r['metadata']['category']})" for r in results])
    prompt = f"Context:\n{context}\n\nQuestion: {query_text}\nAnswer:"
    answer = generator_fn(context, query_text, prompt)
    log_action("RAG", "Retrieved and answered", {"query": query_text, "top_k": top_k})
    return answer, results
