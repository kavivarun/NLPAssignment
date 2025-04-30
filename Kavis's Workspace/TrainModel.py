#!/usr/bin/env python
# ===============================================================
# Minimal end-to-end fine-tune + evaluation for grammar correction
# with automatic split detection for datasets like JFLEG.
# ===============================================================

import os
import math
import tempfile
import warnings
import json

import torch
import evaluate
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)

warnings.filterwarnings("ignore")

# ================== CONFIG (edit here) =========================
MODEL_ID   = "vennify/t5-base-grammar-correction"
OUTPUT_DIR = "runs/t5_ft_gec"
NUM_EPOCHS = 3
BATCH_SIZE = 8
DATASET_ID = "jfleg"            # HF dataset; e.g. "jfleg", "beaufort", etc.
MAX_LEN    = 128
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
# ===============================================================

# ---------- Load dataset and detect splits ----------------------
raw_all = load_dataset(DATASET_ID)       # loads all available splits
splits  = list(raw_all.keys())

if {"train", "validation"}.issubset(splits):
    # Typical case: dataset already has train/validation splits
    raw = DatasetDict({
        "train":      raw_all["train"],
        "validation": raw_all["validation"],
        "test":       raw_all.get("test", raw_all["validation"].select(range(100)))
    })
else:
    # E.g., JFLEG has only 'validation' + 'test'
    print("⚠ No explicit 'train' split—using 90% of 'validation' for train.")
    valid = raw_all["validation"]
    n     = len(valid)
    k     = int(0.9 * n)
    raw = DatasetDict({
        "train":      valid.select(range(k)),
        "validation": valid.select(range(k, n)),
        "test":       raw_all["test"]
    })

# ---------- Tokenizer & Model -----------------------------------
tok   = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID).to(DEVICE)

# ---------- Tokenization function -------------------------------
def tokenize_batch(batch):
    src = batch["sentence"] if "sentence" in batch else batch.get("input", "")
    # For JFLEG, references are lists; take first correction
    tgt = batch.get("corrections", batch.get("target", ""))
    if isinstance(tgt, list):
        tgt = tgt[0]
    return tok(src, text_target=tgt, truncation=True, max_length=MAX_LEN)

data = raw.map(tokenize_batch, batched=True) \
          .remove_columns(raw["train"].column_names)

# ---------- Data collator ---------------------------------------
collator = DataCollatorForSeq2Seq(tok, model=model)

# ---------- Metric loaders --------------------------------------
bleu  = evaluate.load("bleu")
rouge = evaluate.load("rouge")
gleu  = evaluate.load("google_bleu")

try:
    import errant
    ERRANT = errant
except ImportError:
    ERRANT = None
    print("⚠ ERRANT not installed; skipping F0.5 metric.")

def m2_from_pairs(srcs, hyps):
    lines = []
    for s, h in zip(srcs, hyps):
        lines.append(f"S {s}")
        if s != h:
            lines.append(f"A -1 -1|||REPLACE|||{h}|||REQUIRED|||-NONE-|||0")
        lines.append("")
    return "\n".join(lines)

def compute_metrics(eval_pred):
    pred_ids, labels = eval_pred
    preds = tok.batch_decode(pred_ids, skip_special_tokens=True)
    refs  = tok.batch_decode(labels, skip_special_tokens=True)

    out = {
        "BLEU": bleu.compute(
            predictions=preds,
            references=[[r] for r in refs]
        )["bleu"],
        "ROUGE_L": rouge.compute(
            predictions=preds,
            references=refs
        )["rougeL"],
        "GLEU": gleu.compute(
            predictions=preds,
            references=refs
        )["google_bleu"]
    }

    if ERRANT:
        # Compute edit-level F0.5 via ERRANT
        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as hypf, \
             tempfile.NamedTemporaryFile(mode="w+", delete=False) as srcf:
            hypf.write(m2_from_pairs(refs, preds)); hypf.flush()
            srcf.write(m2_from_pairs(refs, refs)); srcf.flush()
            scorer = ERRANT.compare.Compare()
            P, R, F = scorer.run(srcf.name, hypf.name)
            out["F0.5"] = F

    # Round for readability
    return {k: round(v, 4) for k, v in out.items()}

# ---------- Trainer setup ---------------------------------------
training_args = Seq2SeqTrainingArguments(
    OUTPUT_DIR,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    predict_with_generate=True,
    fp16=torch.cuda.is_available()
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=data["train"],
    eval_dataset=data["validation"],
    tokenizer=tok,
    data_collator=collator,
    compute_metrics=compute_metrics
)

# ---------- Train & Save ----------------------------------------
trainer.train()
trainer.save_model(OUTPUT_DIR)

# ---------- Final Test Evaluation ------------------------------
test_metrics = trainer.evaluate(eval_dataset=data["test"])
print("\nFinal test metrics:")
print(json.dumps(test_metrics, indent=2))
