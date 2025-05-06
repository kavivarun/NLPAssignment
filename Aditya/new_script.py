import os
import torch
import numpy as np
import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm
import gc  # For garbage collection
import time

# ----------------------------
# PREPROCESSING FUNCTIONS
# ----------------------------
def preprocess_texts(texts):
    return [" ".join(t.strip().split()) for t in texts]

# ----------------------------
# CONFIGURATION & MODELS
# ----------------------------
GRAMMAR_MODELS = [
    "google/flan-t5-small",
    "facebook/bart-base"
]
STYLE_MODELS = [
    "google/flan-t5-base",
    "sshleifer/distilbart-cnn-12-6"
]

# Using CPU
DEVICE = "cpu"
print("Using CPU for processing")
    
BATCH_SIZE = 1  # Process one at a time to save memory
MAX_LENGTH = 512

# ----------------------------
# MEMORY EFFICIENT MODEL FUNCTIONS
# ----------------------------
def process_with_model(model, tokenizer, text, is_grammar=False, is_t5=False):
    """Process a single text through model in a memory-efficient way"""
    
    # Choose appropriate task prefix based on model type and task
    if is_t5:
        if is_grammar:
            input_text = f"grammar correction: {text}"
        else:
            input_text = f"paraphrase and simplify: {text}"
    else:
        input_text = text
    
    # Tokenize single text
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=MAX_LENGTH,
            min_length=10,
            num_beams=4,
            do_sample=True,  # Enable some randomness for better paraphrasing
            temperature=0.7,  # Add controlled randomness
            top_p=0.9,        # Nucleus sampling
            early_stopping=True
        )
        
    # Decode - explicitly skip special tokens
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Clean up any remaining task prefixes that might have leaked into the output
    if is_t5:
        prefixes = ["grammar correction:", "paraphrase and simplify:", "paraphrase:", "simplify:"]
        for prefix in prefixes:
            if decoded.lower().startswith(prefix):
                decoded = decoded[len(prefix):].strip()
    
    # Free memory
    del inputs, outputs
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return decoded

# ----------------------------
# HUMAN‑EVAL CSV GENERATOR
# ----------------------------
def generate_human_eval_csv(input_csv, text_column, output_csv,):
    """Read a CSV, run every Grammar→Style combo on each row, and write a new CSV
    with side‑by‑side outputs ready for human scoring.
    """
    start_time = time.time()
    print(f"Loading input CSV from {input_csv}")
    df_in = pd.read_csv(input_csv)
    if text_column not in df_in.columns:
        raise KeyError(f"Column '{text_column}' not found in {input_csv}.")
    
    raw_texts = df_in[text_column].astype(str).tolist()
    num_texts = len(raw_texts)
    print(f"Processing {num_texts} texts")
    
    # Pre-clean once
    clean_texts = preprocess_texts(raw_texts)

    # Build output DataFrame starting with the original text
    df_out = pd.DataFrame({text_column: raw_texts})
    
    # Process and intermediate save feature to prevent data loss
    save_interval = max(1, min(10, num_texts // 10))  # Save every ~10% of progress

    # For each grammar model
    for g_idx, gm_name in enumerate(GRAMMAR_MODELS):
        # Load grammar model
        print(f"\n[{g_idx+1}/{len(GRAMMAR_MODELS)}] Loading grammar model: {gm_name}")
        g_tokenizer = AutoTokenizer.from_pretrained(gm_name)
        g_model = AutoModelForSeq2SeqLM.from_pretrained(gm_name)
        is_t5_grammar = "t5" in gm_name.lower()
        
        # Process texts through grammar model
        print(f"Running grammar correction with {gm_name}")
        grammar_outputs = []
        
        for i, text in enumerate(tqdm(clean_texts, desc=f"Grammar model {g_idx+1}/{len(GRAMMAR_MODELS)}")):
            try:
                corrected = process_with_model(g_model, g_tokenizer, text, is_grammar=True, is_t5=is_t5_grammar)
                grammar_outputs.append(corrected)
                
                # Intermediate save to prevent data loss
                if (i + 1) % save_interval == 0:
                    print(f"Progress: {i+1}/{num_texts} texts processed with grammar model")
                    
            except Exception as e:
                print(f"Error processing text {i} with grammar model {gm_name}: {e}")
                # In case of error, keep the original text
                grammar_outputs.append(text)
        
        # Clean up and save intermediate grammar results
        g_clean = preprocess_texts(grammar_outputs)
        grammar_col = f"grammar_{gm_name.split('/')[-1]}"
        df_out[grammar_col] = g_clean
        
        # Save intermediate results
        temp_path = f"intermediate_grammar_{gm_name.split('/')[-1]}.csv"
        df_out.to_csv(temp_path, index=False)
        print(f"Saved intermediate results to {temp_path}")
        
        # Release memory
        del g_model, g_tokenizer
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # For each style model
        for s_idx, sm_name in enumerate(STYLE_MODELS):
            # Load style model
            print(f"\n[{s_idx+1}/{len(STYLE_MODELS)}] Loading style model: {sm_name}")
            s_tokenizer = AutoTokenizer.from_pretrained(sm_name)
            s_model = AutoModelForSeq2SeqLM.from_pretrained(sm_name)
            is_t5_style = "t5" in sm_name.lower()
            
            # Process texts through style model
            print(f"Running style transformation with {gm_name} -> {sm_name}")
            style_outputs = []
            
            for i, text in enumerate(tqdm(g_clean, desc=f"Style model {s_idx+1}/{len(STYLE_MODELS)}")):
                try:
                    styled = process_with_model(s_model, s_tokenizer, text, is_grammar=False, is_t5=is_t5_style)
                    style_outputs.append(styled)
                    
                    # Intermediate save
                    if (i + 1) % save_interval == 0:
                        print(f"Progress: {i+1}/{num_texts} texts processed with style model")
                        
                except Exception as e:
                    print(f"Error processing text {i} with style model {sm_name}: {e}")
                    # In case of error, keep the grammar-corrected text
                    style_outputs.append(text)
            
            # Save results
            col_name = f"{gm_name.split('/')[-1]}__{sm_name.split('/')[-1]}"
            df_out[col_name] = style_outputs
            
            # Save intermediate results
            temp_path = f"intermediate_{gm_name.split('/')[-1]}_{sm_name.split('/')[-1]}.csv"
            df_out.to_csv(temp_path, index=False)
            print(f"Saved intermediate results to {temp_path}")
            
            # Release memory
            del s_model, s_tokenizer
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Add empty score columns
    for gm in GRAMMAR_MODELS:
        gm_short = gm.split('/')[-1]
        for sm in STYLE_MODELS:
            sm_short = sm.split('/')[-1]
            df_out[f"score_{gm_short}__{sm_short}"] = ""
    
    # Save final output
    df_out.to_csv(output_csv, index=False)
    
    # Print execution summary
    end_time = time.time()
    total_time = end_time - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\nProcessing completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print(f"Final results saved to {output_csv}")
    
    # Cleanup intermediate files
    for gm in GRAMMAR_MODELS:
        gm_short = gm.split('/')[-1]
        if os.path.exists(f"intermediate_grammar_{gm_short}.csv"):
            os.remove(f"intermediate_grammar_{gm_short}.csv")
        for sm in STYLE_MODELS:
            sm_short = sm.split('/')[-1]
            if os.path.exists(f"intermediate_{gm_short}_{sm_short}.csv"):
                os.remove(f"intermediate_{gm_short}_{sm_short}.csv")
    print("Intermediate files cleaned up")
    return df_out

# ----------------------------
# MAIN FUNCTION
# ----------------------------
def main():
    print("\n====== PARAPHRASING PIPELINE ======")
    print("This script will process the dataset through grammar and style models")
    print("Processing may take several hours depending on dataset size")
    print("Intermediate results will be saved regularly to prevent data loss")
    
    try:
        # Small test to verify torch is working
        x = torch.randn(1, 3)
        print("✓ PyTorch working properly")
        
        generate_human_eval_csv(
            os.path.join("files", "dirty_data.csv"),  # Input CSV with poorly written texts
            'Text',                   # Column name containing the text
            os.path.join("files", "human_eval_outputs.csv") # Output CSV with all paraphrased versions
        )
    except Exception as e:
        print(f"\n❌ Error encountered: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()