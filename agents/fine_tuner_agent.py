# agents/fine_tuner_agent.py
import os
from crewai import Agent
from database.logger import log_action
from dotenv import load_dotenv

import pandas as pd
from datasets import Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "google/flan-t5-base")

# Directory where labeled or Q&A data is stored
LABELED_DIR = "datasets/labeled_data"
QA_DIR = "datasets/qa_data"
FINETUNE_DIR = "models/fine_tuned"

os.makedirs(FINETUNE_DIR, exist_ok=True)

# -------------------------------
# Helper: prepare dataset for fine-tuning
# -------------------------------
def load_training_data():
    """
    Collects Q&A pairs from your labeled CSV or QA JSONs.
    Expect JSON lines: [{"question": "...", "answer": "..."}]
    """
    data = []
    # 1️⃣ Load any Q&A JSON files
    if os.path.exists(QA_DIR):
        for f in os.listdir(QA_DIR):
            if f.endswith(".json"):
                df = pd.read_json(os.path.join(QA_DIR, f))
                for _, row in df.iterrows():
                    if "question" in row and "answer" in row:
                        data.append({
                            "input_text": row["question"],
                            "target_text": row["answer"]
                        })

    if not data:
        print("[fine_tuner_agent] No Q&A data found for fine-tuning.")
        return None

    print(f"[fine_tuner_agent] Loaded {len(data)} samples for training.")
    return Dataset.from_list(data)


# -------------------------------
# Core Fine-tuning Function
# -------------------------------
def fine_tune_local_model(mode="auto"):
    """
    mode = "auto" → triggered by low review accuracy (fine-tune on combined_qa.json)
    mode = "global" → user-triggered retraining on all data
    """
    print(f"[fine_tuner_agent] Starting fine-tuning mode: {mode}")
    dataset = load_training_data()   # this already loads all QA files
    if not dataset:
        print("[fine_tuner_agent] No data available for fine-tuning.")
        return None

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    # LoRA configuration (lightweight fine-tuning)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q", "v"],  # T5 attention submodules
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_2_SEQ_LM"
    )

    model = get_peft_model(model, lora_config)

    def preprocess(examples):
        inputs = examples["input_text"]
        targets = examples["target_text"]
        model_inputs = tokenizer(
            inputs, max_length=256, truncation=True, padding="max_length"
        )
        labels = tokenizer(
            targets, max_length=128, truncation=True, padding="max_length"
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_dataset = dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)

    training_args = TrainingArguments(
        output_dir=FINETUNE_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        learning_rate=5e-5,
        logging_dir=os.path.join(FINETUNE_DIR, "logs"),
        save_total_limit=1,
        save_strategy="epoch",
        remove_unused_columns=False,
        push_to_hub=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    print("[fine_tuner_agent] Starting fine-tuning...")
    trainer.train()
    model.save_pretrained(FINETUNE_DIR)
    tokenizer.save_pretrained(FINETUNE_DIR)

    log_action("FineTuner Agent", "Model fine-tuned", {"samples": len(dataset), "model": MODEL_NAME})
    print(f"[fine_tuner_agent] ✅ Fine-tuning complete! Model saved to {FINETUNE_DIR}")

    return FINETUNE_DIR


# -------------------------------
# CrewAI Agent Wrapper
# -------------------------------
fine_tuner_agent = Agent(
    role="FineTuner Agent",
    goal="Perform LoRA fine-tuning on labeled or Q&A data using flan-t5-base.",
    backstory="This agent fine-tunes the local model incrementally as new labeled or Q&A data appears.",
    llm=None
)