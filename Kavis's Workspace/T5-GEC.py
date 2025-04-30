import torch
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import (
    T5ForConditionalGeneration,
    T5TokenizerFast,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments
)
import evaluate

# === 1. Load WikiAuto dataset (manual config) ===
dataset = load_dataset("chaojiang06/wiki_auto", "manual")  # contains only 'train' and 'test'

# === 2. Load tokenizer and model ===
tokenizer = T5TokenizerFast.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained("t5-base")

# === 3. Preprocessing ===
max_input_length = 128
max_target_length = 128

def preprocess(examples):
    inputs = ["grammar: " + s for s in examples["simple_sentence"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, padding="max_length")
    labels = tokenizer(text_target=examples["normal_sentence"], max_length=max_target_length, truncation=True, padding="max_length")
    model_inputs["labels"] = [
        [(token if token != tokenizer.pad_token_id else -100) for token in label]
        for label in labels["input_ids"]
    ]
    return model_inputs

# Apply preprocessing
tokenized = dataset.map(preprocess, batched=True, remove_columns=dataset["train"].column_names)

# Split training into train/validation (90/10)
split = tokenized["train"].train_test_split(test_size=0.1)
train_set = split["train"]
val_set = split["test"]

# Data collator handles padding dynamically
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# === 4. Define metrics (BLEU + GLEU) ===
bleu = evaluate.load("sacrebleu")
gleu = evaluate.load("google_bleu")

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    bleu_score = bleu.compute(predictions=decoded_preds, references=[[l] for l in decoded_labels])["score"]
    gleu_score = gleu.compute(predictions=decoded_preds, references=decoded_labels)["google_bleu"]

    return {"bleu": bleu_score, "gleu": gleu_score}

# === 5. Training configuration ===
training_args = TrainingArguments(
    output_dir="t5-gec-output",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=20,  # works on HF transformers >=3.0.0
    save_total_limit=2
)

# === 6. Trainer setup ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_set,
    eval_dataset=val_set,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# === 7. Train the model ===
trainer.train()

# === 8. Sentence-level evaluation on test set ===
test_dataloader = trainer.get_eval_dataloader(tokenized["test"])
all_preds, all_labels = [], []

for batch in test_dataloader:
    batch = {k: v.to(trainer.args.device) for k, v in batch.items() if k in ["input_ids", "attention_mask"]}
    outputs = model.generate(**batch)
    all_preds.extend(outputs.cpu().numpy())
    all_labels.extend(batch["input_ids"].cpu().numpy())  # for source display

decoded_preds = tokenizer.batch_decode(all_preds, skip_special_tokens=True)
decoded_refs  = tokenizer.batch_decode(tokenized["test"]["labels"], skip_special_tokens=True)
decoded_srcs  = tokenizer.batch_decode(all_labels, skip_special_tokens=True)

# Save sentence-level BLEU + GLEU scores
rows = []
for src, ref, hyp in zip(decoded_srcs, decoded_refs, decoded_preds):
    b = bleu.compute(predictions=[hyp], references=[[ref]])["score"]
    g = gleu.compute(predictions=[hyp], references=[ref])["google_bleu"]
    rows.append({"source": src, "reference": ref, "hypothesis": hyp, "bleu": b, "gleu": g})

df = pd.DataFrame(rows)
df.to_csv("gec_sentence_metrics.csv", index=False)

print("\n✅ Training complete!")
print("📊 Sentence-level metrics saved to: gec_sentence_metrics.csv")
print("📈 Final corpus scores:", compute_metrics((all_preds, tokenized["test"]["labels"])))
