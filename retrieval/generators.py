# retrieval/generators.py
import os
from dotenv import load_dotenv
load_dotenv()
BACKEND = os.getenv("MODEL_BACKEND", "transformers")
MODEL_NAME = os.getenv("MODEL_NAME", "microsoft/phi-3-mini-4k-instruct")

if BACKEND == "ollama":
    from openai import OpenAI
    client = OpenAI(base_url=os.getenv("OPENAI_API_BASE"), api_key=os.getenv("OPENAI_API_KEY"))
    def generator_fn(context, query, prompt=None):
        if prompt is None:
            prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL"),
            messages=[{"role":"system","content":"You are a helpful financial assistant."},{"role":"user","content":prompt}],
            temperature=0.1
        )
        return resp.choices[0].message.content.strip()
else:
    from core.model_loader import load_phi3_model, generate_with_model
    model, tokenizer = load_phi3_model()
    def generator_fn(context, query, prompt=None):
        if prompt is None:
            prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer in 1-2 sentences:"
        return generate_with_model(model, tokenizer, prompt, max_new_tokens=200)
