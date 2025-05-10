import os
import time
import gc
import numpy as np
from tqdm import tqdm
import re
import torch
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, Optional, Tuple

# Dictionary to store loaded models (only load when needed)
grammar_pipes = {}
style_pipes = {}
coedit_objs = {}  

# Fixed model choices
GRAMMAR_MODEL = "facebook/bart-base"
STYLE_MODEL = "rajistics/informal_formal_style_transfer"
COEDIT_MODEL  = "grammarly/coedit-large"  

# Device selection - use GPU if available, otherwise CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")

BATCH_SIZE = 1 
MAX_LENGTH = 512

def load_coedit_if_needed(model_name: str = COEDIT_MODEL):
    """Lazy-load Grammarly CoEdit (tokenizer, model) on the selected device."""
    from transformers import AutoTokenizer, T5ForConditionalGeneration

    if model_name in coedit_objs:
        return coedit_objs[model_name]

    print(f"Loading CoEdit model: {model_name} on {DEVICE}")
    tok   = AutoTokenizer.from_pretrained(model_name)
    mdl   = T5ForConditionalGeneration.from_pretrained(model_name).to(DEVICE)
    coedit_objs[model_name] = (tok, mdl)
    return tok, mdl

def coedit_paraphrase(text: str, max_tokens: int = 512) -> str:
    """
    Paraphrase / improve long text with Grammarly CoEdit.
    Splits into sentence chunks that fit the model context window.
    """
    import nltk
    from nltk.tokenize import sent_tokenize
    try:                # make sure punkt is available
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)

    tok, mdl = load_coedit_if_needed()
    sentences = sent_tokenize(text)

    chunks, current = [], ""
    for sent in sentences:
        n_tokens = tok(current + " " + sent, return_tensors="pt").input_ids.shape[1]
        if n_tokens < max_tokens:
            current += " " + sent
        else:
            chunks.append(current.strip())
            current = sent
    if current:
        chunks.append(current.strip())

    improved = []
    for chunk in chunks:
        prompt  = f"Paraphrase and improve the clarity, style, and grammar of the following text: {chunk}"
        inputs  = tok(prompt, return_tensors="pt", truncation=True, max_length=max_tokens).to(DEVICE)
        with torch.no_grad():
            outs = mdl.generate(
                inputs.input_ids,
                max_length=max_tokens,
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
            )
        improved.append(tok.decode(outs[0], skip_special_tokens=True))

    return " ".join(improved).strip()

def run_coedit_model_async(input_text: str) -> Tuple[str, Optional[str]]:
    """Thread wrapper for CoEdit paraphraser."""
    try:
        result = coedit_paraphrase(input_text)
        return result, None
    except Exception as e:
        return None, str(e)

def load_model_if_needed(model_name, is_grammar=True):
    """Lazy loading of models only when needed with GPU support"""
    from transformers import pipeline
    
    global grammar_pipes, style_pipes
    
    # Check if model is already loaded
    target_dict = grammar_pipes if is_grammar else style_pipes
    if model_name in target_dict:
        print(f"Model {model_name} already loaded on {target_dict[model_name].device}")
        return target_dict[model_name]
    
    # Load the model if not found
    print(f"Loading {'grammar' if is_grammar else 'style'} model: {model_name} on {DEVICE}")
    
    # Get torch device index for pipeline
    device_idx = 0 if torch.cuda.is_available() else -1  # 0 for GPU, -1 for CPU
    
    if is_grammar:
        pipe = pipeline(
            "text2text-generation",
            model=model_name,
            device=device_idx,
            do_sample=False,
            num_beams=4,
            max_length=MAX_LENGTH,
            model_kwargs={"torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32}
        )
        grammar_pipes[model_name] = pipe
    else:
        pipe = pipeline(
            "text2text-generation",
            model=model_name,
            device=device_idx,
            do_sample=True,
            temperature=0.9,
            top_p=0.8,
            num_beams=4,
            max_length=int(MAX_LENGTH / 2),
            model_kwargs={"torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32}
        )
        style_pipes[model_name] = pipe
    
    print(f"Successfully loaded {model_name} on device: {pipe.device}")
    return pipe

def preprocess_texts(texts):
    """Clean and preprocess text"""
    return [" ".join(t.strip().split()) for t in texts]

def run_pipe_with_prefix(
    pipe,
    texts: list[str],
    model_name: str,
    is_grammar: bool,
    batch_size: int = BATCH_SIZE,
    max_len: int = MAX_LENGTH,
):
    """Run text through a pipeline with appropriate prefix"""
    
    # Determine model family → choose prefixes
    t5 = "t5" in model_name.lower()
    if t5:
        gram_pref  = "Fix all grammar and spelling errors> "
        style_pref = "Paraphase and Simplify> "
    else:
        gram_pref  = "Fix all grammar and spelling errors> "
        style_pref = "Paraphase and Simplify> "

    prefix = gram_pref if is_grammar else style_pref
    strip_prefixes = (gram_pref, style_pref, "Paraphrase: ", "Grammar correction: ")

    outputs_all = []
    
    for i in tqdm(
        range(0, len(texts), batch_size),
        desc=f"{'Grammar' if is_grammar else 'Style'} pass ({model_name}) on {pipe.device}",
        leave=False,
    ):
        chunk = texts[i : i + batch_size]

        # 1) add prefix
        inp_with_pref = [prefix + txt for txt in chunk]

        # 2) pipeline call
        outs = pipe(
            inp_with_pref,
            max_length=max_len,
            batch_size=batch_size,
        )

        # 3) decode & strip prefixes
        for out in outs:
            decoded = out["generated_text"].strip()
            for p in strip_prefixes:
                if decoded.startswith(p):
                    decoded = decoded[len(p):].lstrip()
            outputs_all.append(decoded)

    return outputs_all

def evaluate_text_quality(original_text: str, processed_text: str):
    """
    Compare processed_text to original_text and return a metrics dict.
    1) Cleans and splits overly long runs.
    2) Chunks into sentences and computes per-sentence readability.
    3) Averages the FKGL and FRE across sentences.
    4) Falls back to counts-only if textstat is missing or no sentences found.
    """
    # ---------- helpers ----------
    _BULLET          = re.compile(r'[\u2022\u2023\u25E6\u2043\u2219]')
    _TERM            = re.compile(r'([.!?])(\s|$)')
    _LONG_SENT_SPLIT = re.compile(r'\b(?:and|but|so)\b|,')
    _SENT_SPLIT      = re.compile(r'[.!?]')

    def _clean(txt: str) -> str:
        txt = _BULLET.sub('.', txt)
        txt = txt.replace('\n', '. ')
        txt = _TERM.sub(r'\1 ', txt)
        txt = re.sub(r'\.{2,}', '.', txt)
        txt = re.sub(r'\s{2,}', ' ', txt)
        return txt.strip()

    def _split_long_runs(txt: str, max_len: int = 40) -> str:
        words = txt.split()
        if len(words) <= max_len:
            return txt
        out, chunk = [], []
        for w in words:
            chunk.append(w)
            if len(chunk) >= max_len and _LONG_SENT_SPLIT.search(w):
                out.append(' '.join(chunk) + '.')
                chunk = []
        out.append(' '.join(chunk))
        return ' '.join(out)

    def _prep(txt: str) -> str:
        return _clean(_split_long_runs(txt))

    def _basic_counts(txt: str):
        words = txt.split()
        sentences = [s for s in _SENT_SPLIT.split(txt) if s.strip()]
        return len(words), len(sentences)

    def _sentence_list(txt: str):
        # split into sentences, strip out empties
        return [s.strip() for s in _SENT_SPLIT.split(txt) if s.strip()]

    # ---------- prepare texts ----------
    orig = _prep(original_text)
    proc = _prep(processed_text)

    # ---------- initialize all metrics ----------
    metrics = {
        "fkgl_original": None,
        "fkgl_processed": None,
        "fkgl_change": None,
        "fre_original": None,
        "fre_processed": None,
        "fre_change": None,
    }

    # ---------- attempt readability via textstat ----------
    try:
        import textstat

        orig_sents = _sentence_list(orig)
        proc_sents = _sentence_list(proc)
        if not orig_sents or not proc_sents:
            raise ValueError("No sentences detected after cleaning.")

        # compute per-sentence scores
        fkgl_o_list = [textstat.flesch_kincaid_grade(s) for s in orig_sents]
        fkgl_p_list = [textstat.flesch_kincaid_grade(s) for s in proc_sents]
        fre_o_list  = [textstat.flesch_reading_ease(s)   for s in orig_sents]
        fre_p_list  = [textstat.flesch_reading_ease(s)   for s in proc_sents]

        # average
        avg_fkgl_o = sum(fkgl_o_list) / len(fkgl_o_list)
        avg_fkgl_p = sum(fkgl_p_list) / len(fkgl_p_list)
        avg_fre_o  = sum(fre_o_list)  / len(fre_o_list)
        avg_fre_p  = sum(fre_p_list)  / len(fre_p_list)

        metrics.update({
            "fkgl_original": avg_fkgl_o,
            "fkgl_processed": avg_fkgl_p,
            "fkgl_change": avg_fkgl_o - avg_fkgl_p,
            "fre_original": avg_fre_o,
            "fre_processed": avg_fre_p,
            "fre_change": avg_fre_p - avg_fre_o,
        })

    except Exception:
        metrics["note"] = (
            "textstat unavailable or input lacked sentences; "
            "readability metrics set to None."
        )

    # ---------- always add counts ----------
    wo, so = _basic_counts(orig)
    wp, sp = _basic_counts(proc)
    metrics.update({
        "word_count_original": wo,
        "word_count_processed": wp,
        "word_count_change": wp - wo,
        "sentence_count_original": so,
        "sentence_count_processed": sp,
        "sentence_count_change": sp - so,
    })

    return metrics

def display_text_quality_metrics(original_text, processed_text, title="Text Quality Metrics", container=None):
    """
    Calculate and display text quality metrics in a Streamlit container
    
    Args:
        original_text (str): The original text before processing
        processed_text (str): The processed/improved text
        title (str): Title for the metrics section
        container: Streamlit container to display metrics in (default: st)
    
    Returns:
        dict: The calculated metrics dictionary
    """
    # Import streamlit only when needed
    import streamlit as st
    
    # Use provided container or default to st
    if container is None:
        container = st
    
    # Calculate metrics
    metrics = evaluate_text_quality(original_text, processed_text)
    
    if metrics:
        with container.expander(title):
            # Add device info
            container.info(f"Models are running on: {DEVICE}")
            
            # Create 3 columns for metrics display
            col1, col2, col3 = container.columns(3)
            
            # Display readability metrics if available
            if "fkgl_processed" in metrics:
                with col1:
                    container.metric(
                        "Readability Grade Level", 
                        f"{metrics['fkgl_processed']:.1f}",
                        f"{-metrics['fkgl_change']:.1f}"
                    )
                
                with col2:
                    container.metric(
                        "Reading Ease", 
                        f"{metrics['fre_processed']:.1f}",
                        f"{metrics['fre_change']:.1f}"
                    )
                
                with col3:
                    container.metric(
                        "Words", 
                        f"{metrics['word_count_processed']}",
                        f"{metrics['word_count_change']}"
                    )
                
                # Create another row of metrics
                col1, col2, col3 = container.columns(3)
                
                with col1:
                    container.metric(
                        "Sentences", 
                        f"{metrics['sentence_count_processed']:.0f}",
                        f"{metrics['sentence_count_change']:.0f}"
                    )
                
                with col2:
                    # Words per sentence
                    if metrics['sentence_count_processed'] > 0:
                        wps_orig = metrics['word_count_original'] / metrics['sentence_count_original'] if metrics['sentence_count_original'] > 0 else 0
                        wps_proc = metrics['word_count_processed'] / metrics['sentence_count_processed']
                        container.metric(
                            "Words per Sentence", 
                            f"{wps_proc:.1f}",
                            f"{wps_proc - wps_orig:.1f}"
                        )
                
                container.caption("**Reading metrics guide:**  \n"
                                "- **Grade Level**: Lower = easier to read (1-20 scale)  \n"
                                "- **Reading Ease**: Higher = easier to read (0-100 scale)  \n"
                                "- **Words per Sentence**: Lower generally means clearer writing")
            else:
                # Display basic metrics if textstat metrics weren't available
                with col1:
                    container.metric(
                        "Word Count", 
                        f"{metrics['word_count_processed']}",
                        f"{metrics['word_count_change']}"
                    )
                
                with col2:
                    container.metric(
                        "Character Count", 
                        f"{metrics['char_count_processed']}",
                        f"{metrics['char_count_change']}"
                    )
                
                with col3:
                    container.metric(
                        "Sentence Count", 
                        f"{metrics['sentence_count_processed']}",
                        f"{metrics['sentence_count_change']}"
                    )
                
                container.caption("Metrics shown are basic counts as textstat library is not available. "
                                "Install textstat for more detailed readability metrics.")
    
    return metrics

# Add parallel processing functions
def run_grammar_model_async(input_text: str) -> Tuple[str, None]:
    """Run grammar model in a separate thread"""
    try:
        clean_texts = preprocess_texts([input_text])
        g_pipe = load_model_if_needed(GRAMMAR_MODEL, is_grammar=True)
        g_fixed = run_pipe_with_prefix(
            g_pipe, clean_texts, GRAMMAR_MODEL, is_grammar=True
        )
        return g_fixed[0], None
    except Exception as e:
        return None, str(e)

def run_style_model_async(grammar_corrected_text: str) -> Tuple[str, None]:
    """Run style model in a separate thread"""
    try:
        g_clean = preprocess_texts([grammar_corrected_text])
        s_pipe = load_model_if_needed(STYLE_MODEL, is_grammar=False)
        preds = run_pipe_with_prefix(
            s_pipe, g_clean, STYLE_MODEL, is_grammar=False
        )
        return preds[0], None
    except Exception as e:
        return None, str(e)

def run_translation_model_async(input_text: str) -> Tuple[str, None]:
    """Run translation model in a separate thread"""
    try:
        result = test_with_translation_models(input_text)
        return result, None
    except Exception as e:
        return None, str(e)

def run_ollama_models_async(input_text: str) -> Tuple[Dict[str, Any], None]:
    """Run Ollama models in a separate thread"""
    try:
        from Ollama_client import chat_with_models
        base_prompt = f"Please fix grammatical errors in this sentence and improve its style: {input_text}. Add it between `<fixg>` and `</fixg>` tags."
        data = chat_with_models(base_prompt, ["mistral:latest"])
        return data, None
    except Exception as e:
        return None, str(e)

def process_text_parallel(input_text: str, progress_callback=None) -> Dict[str, Any]:
    """
    Process text with all models in parallel for improved performance
    
    Args:
        input_text (str): Text to process
        progress_callback: Optional callback function to update progress
    
    Returns:
        Dict containing all model results
    """
    results = {
        "grammar_corrected": None,
        "style_improved": None,
        "coedit_result":    None,
        "translation": None,
        "ollama_results": None,
        "errors": []
    }
    
    try:
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Start all models in parallel
            futures = {}
            
            # Start grammar correction first
            grammar_future = executor.submit(run_grammar_model_async, input_text)
            futures["grammar"] = grammar_future

            coedit_future = executor.submit(run_coedit_model_async, input_text)
            futures["coedit"] = coedit_future
            
            # Start translation model (independent of grammar)
            translation_future = executor.submit(run_translation_model_async, input_text)
            futures["translation"] = translation_future
            
            # Start Ollama models (independent of other models)
            ollama_future = executor.submit(run_ollama_models_async, input_text)
            futures["ollama"] = ollama_future
            
            # Process grammar result when ready
            grammar_result, grammar_error = grammar_future.result()
            if grammar_error:
                results["errors"].append(f"Grammar model error: {grammar_error}")
            else:
                results["grammar_corrected"] = grammar_result
                
                # Start style model once grammar is done
                style_future = executor.submit(run_style_model_async, grammar_result)
                futures["style"] = style_future
                
                # Get style result
                style_result, style_error = style_future.result()
                if style_error:
                    results["errors"].append(f"Style model error: {style_error}")
                else:
                    results["style_improved"] = style_result

            coedit_result, coedit_error = coedit_future.result()
            if coedit_error:
                results["errors"].append(f"CoEdit model error: {coedit_error}")
            else:
                results["coedit_result"] = coedit_result
            
            # Get translation result
            translation_result, translation_error = translation_future.result()
            if translation_error:
                results["errors"].append(f"Translation model error: {translation_error}")
            else:
                results["translation"] = translation_result
            
            # Get Ollama results
            ollama_result, ollama_error = ollama_future.result()
            if ollama_error:
                results["errors"].append(f"Ollama models error: {ollama_error}")
            else:
                results["ollama_results"] = ollama_result
            
            # Update progress if callback provided
            if progress_callback:
                progress_callback(100)
                
    except Exception as e:
        results["errors"].append(f"Parallel processing error: {str(e)}")
    
    # Run garbage collection to free memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return results

def paraphrase_single_text(input_text):
    """
    Original sequential processing function for backward compatibility
    """
    try:
        # Show current device usage
        print(f"Processing on device: {DEVICE}")
        
        clean_texts = preprocess_texts([input_text])
        
        # Load and run grammar model
        g_pipe = load_model_if_needed(GRAMMAR_MODEL, is_grammar=True)
        g_fixed = run_pipe_with_prefix(
            g_pipe, clean_texts, GRAMMAR_MODEL, is_grammar=True
        )
        g_clean = preprocess_texts(g_fixed)
        
        # Load and run style model
        s_pipe = load_model_if_needed(STYLE_MODEL, is_grammar=False)
        preds = run_pipe_with_prefix(
            s_pipe, g_clean, STYLE_MODEL, is_grammar=False
        )
        
        # Run garbage collection to free memory (important on GPU)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return preds[0], g_fixed[0]  # Return both the final result and grammar-corrected text
        
    except Exception as e:
        print(f"Error during paraphrasing: {e}")
        import traceback
        traceback.print_exc()
        return f"Error during paraphrasing: {str(e)}", input_text

def save_text_to_file(text, file_path):
    """Save text to a file"""
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text)
        return True
    except Exception as e:
        print(f"Error saving text to file: {e}")
        return False

def test_with_translation_models(bad_text):
    from transformers import (
        AutoTokenizer, 
        AutoModelForCausalLM, 
        MarianMTModel, 
        MarianTokenizer, 
        MBartForConditionalGeneration, 
        MBart50TokenizerFast
    )
    import torch

    # Ensure we're using the same device
    device = DEVICE
    
    print(f"\n--- MarianMT (en→fr→en) Paraphrasing on {device} ---")
    
    # English to French
    en_fr_model_name = "Helsinki-NLP/opus-mt-en-fr"
    fr_en_model_name = "Helsinki-NLP/opus-mt-fr-en"
    
    try:
        en_fr_tokenizer = MarianTokenizer.from_pretrained(en_fr_model_name)
        en_fr_model = MarianMTModel.from_pretrained(en_fr_model_name).to(device)
        fr_en_tokenizer = MarianTokenizer.from_pretrained(fr_en_model_name)
        fr_en_model = MarianMTModel.from_pretrained(fr_en_model_name).to(device)

        # English to French
        fr_tokens = en_fr_tokenizer([bad_text], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            fr_translation = en_fr_model.generate(**fr_tokens)
        fr_text = en_fr_tokenizer.batch_decode(fr_translation, skip_special_tokens=True)[0]

        # French back to English
        en_tokens = fr_en_tokenizer([fr_text], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            en_translation = fr_en_model.generate(**en_tokens)
        marian_result = fr_en_tokenizer.batch_decode(en_translation, skip_special_tokens=True)[0]
        
        print(f"Translation completed on {device}")
        
        # Clean up GPU memory if using GPU
        if torch.cuda.is_available():
            del en_fr_model, fr_en_model, fr_tokens, en_tokens
            torch.cuda.empty_cache()
        
        return marian_result
        
    except Exception as e:
        print(f"Error in translation: {e}")
        return None

def extract_fixed_text(text):
    """
    Extract text between <fixg> and </fixg> tags
    
    Args:
        text: String containing the model response
        
    Returns:
        Extracted text or None if not found
    """
    pattern = r'<fixg>(.*?)</fixg>'
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    else:
        return None

# Example usage
def extract_from_model_responses(data):
    results = {}
    
    for model_name, result in data.items():
        if result['status'] == 'success':
            response_text = result['response']['message']['content']
            fixed_text = extract_fixed_text(response_text)
            results[model_name] = fixed_text
        else:
            results[model_name] = f"Error: {result.get('error', 'Unknown error')}"
    
    return results

# Check GPU memory usage (if available)
def check_gpu_memory():
    """Check and display GPU memory usage"""
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        gpu_allocated = torch.cuda.memory_allocated(0)
        gpu_cached = torch.cuda.memory_reserved(0)
        
        print(f"GPU Memory - Total: {gpu_memory/1024**3:.2f}GB, Allocated: {gpu_allocated/1024**3:.2f}GB, Cached: {gpu_cached/1024**3:.2f}GB")
        return {
            "total": gpu_memory,
            "allocated": gpu_allocated,
            "cached": gpu_cached,
            "free": gpu_memory - gpu_cached
        }
    else:
        print("No GPU available")
        return None

# Clear GPU cache
def clear_gpu_cache():
    """Clear GPU cache to free up memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU cache cleared")
    else:
        print("No GPU to clear cache")