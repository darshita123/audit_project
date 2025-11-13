from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch, os

def load_phi3_model():
    model_name = os.getenv("MODEL_NAME", "google/flan-t5-base")
    print(f"[model_loader] Loading model: {model_name} (device_map=cpu)")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        device_map="cpu",
        torch_dtype=torch.float32
    )
    return model, tokenizer

def generate_with_model(model, tokenizer, prompt, max_new_tokens=64):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)