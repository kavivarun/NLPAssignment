from datasets import load_dataset
from transformers import pipeline
from tqdm.auto import tqdm
import errant, spacy, sacrebleu, subprocess

# 1. Load & rename WikiAuto (as before)
wiki = load_dataset("chaojiang06/wiki_auto", "manual", split="test[:10%]", trust_remote_code=True)
wiki = wiki.rename_column("simple_sentence", "informal") \
           .rename_column("normal_sentence", "formal")

# 2. Init pipelines
nlp       = spacy.load("en_core_web_sm")
annotator = errant.load("en", nlp)

gec_bart = pipeline("text2text-generation",
                    model="gotutiyan/gec-bart-large",
                    device=0)
stylef   = pipeline("text2text-generation",
                    model="rajistics/informal_formal_style_transfer",
                    device=0)

# 3. Prepare for batching
batch_size = 32
num_samples = len(wiki)
num_batches = (num_samples + batch_size - 1) // batch_size

orig_sents, corr_sents, preds, refs = [], [], [], []

# 4. Loop with tqdm for ETA
for start_idx in tqdm(range(0, num_samples, batch_size),
                      total=num_batches,
                      desc="Correct → Formalize",
                      unit="batch"):
    end_idx = min(start_idx + batch_size, num_samples)
    batch = wiki[start_idx:end_idx]
    orig_batch = batch["informal"]
    
    # a) Grammar correction
    gec_out = gec_bart(orig_batch,
                       max_length=128,
                       clean_up_tokenization_spaces=True,
                       batch_size=batch_size)
    corrected = [o["generated_text"] for o in gec_out]

    # b) Formality transfer
    style_out = stylef(corrected,
                       max_length=128,
                       clean_up_tokenization_spaces=True,
                       batch_size=batch_size)
    formalized = [o["generated_text"] for o in style_out]

    # c) Collect
    orig_sents.extend(orig_batch)
    corr_sents.extend(corrected)
    preds.extend(formalized)
    refs.extend(batch["formal"])

# 5. ERRANT scoring (unchanged)
with open("orig.txt","w",encoding="utf8") as f:
    f.write("\n".join(orig_sents))
with open("sys.txt","w",encoding="utf8") as f:
    f.write("\n".join(corr_sents))

subprocess.run([
    "errant_parallel", "-orig","orig.txt","-cor","sys.txt","-out","sys.m2"
], check=True)
# …errant_compare etc.

# 6. BLEU + formality success
bleu = sacrebleu.corpus_bleu(preds, [refs]).score
print(f"\nBLEU = {bleu:.1f}%")
