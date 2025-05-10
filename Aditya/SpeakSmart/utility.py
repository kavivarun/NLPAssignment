import os
import time
import gc
import numpy as np
from tqdm import tqdm

# Dictionary to store loaded models (only load when needed)
grammar_pipes = {}
style_pipes = {}

# Fixed model choices
GRAMMAR_MODEL = "facebook/bart-base"
STYLE_MODEL = "rajistics/informal_formal_style_transfer"

# Using CPU
DEVICE = "cpu"
BATCH_SIZE = 1 
MAX_LENGTH = 512

def load_model_if_needed(model_name, is_grammar=True):
    """Lazy loading of models only when needed"""
    from transformers import pipeline
    
    global grammar_pipes, style_pipes
    
    # Check if model is already loaded
    target_dict = grammar_pipes if is_grammar else style_pipes
    if model_name in target_dict:
        return target_dict[model_name]
    
    # Load the model if not found
    print(f"Loading {'grammar' if is_grammar else 'style'} model: {model_name}")
    if is_grammar:
        pipe = pipeline(
            "text2text-generation",
            model=model_name,
            device=DEVICE,
            do_sample=False,
            num_beams=4,
            max_length=MAX_LENGTH
        )
        grammar_pipes[model_name] = pipe
    else:
        pipe = pipeline(
            "text2text-generation",
            model=model_name,
            device=DEVICE,
            do_sample=True,
            temperature=0.9,
            top_p=0.8,
            num_beams=4,
            max_length=int(MAX_LENGTH / 2)
        )
        style_pipes[model_name] = pipe
    
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
        desc=f"{'Grammar' if is_grammar else 'Style'} pass ({model_name})",
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

def evaluate_text_quality(original_text, processed_text):
    """
    Evaluate the quality of processed text compared to original
    Returns metrics dictionary
    """
    try:
        # Import evaluation libraries here to avoid loading them at startup
        import textstat
        
        # Calculate readability scores
        fkgl_orig = textstat.flesch_kincaid_grade(original_text)
        fkgl_proc = textstat.flesch_kincaid_grade(processed_text)
        fre_orig = textstat.flesch_reading_ease(original_text)
        fre_proc = textstat.flesch_reading_ease(processed_text)
        
        # Calculate word and sentence counts
        word_count_orig = len(original_text.split())
        word_count_proc = len(processed_text.split())
        sentence_count_orig = textstat.sentence_count(original_text)
        sentence_count_proc = textstat.sentence_count(processed_text)
        
        return {
            "fkgl_original": fkgl_orig,
            "fkgl_processed": fkgl_proc,
            "fkgl_change": fkgl_orig - fkgl_proc,
            "fre_original": fre_orig,
            "fre_processed": fre_proc,
            "fre_change": fre_proc - fre_orig,
            "word_count_original": word_count_orig,
            "word_count_processed": word_count_proc,
            "word_count_change": word_count_proc - word_count_orig,
            "sentence_count_original": sentence_count_orig,
            "sentence_count_processed": sentence_count_proc,
            "sentence_count_change": sentence_count_proc - sentence_count_orig
        }
    except ImportError:
        # Calculate basic stats without textstat
        orig_words = len(original_text.split())
        proc_words = len(processed_text.split())
        orig_chars = len(original_text)
        proc_chars = len(processed_text)
        
        # Rough sentence count (not as accurate as textstat)
        orig_sentences = len([s for s in original_text.split('.') if s.strip()])
        proc_sentences = len([s for s in processed_text.split('.') if s.strip()])
        
        return {
            "word_count_original": orig_words,
            "word_count_processed": proc_words,
            "word_count_change": proc_words - orig_words,
            "char_count_original": orig_chars,
            "char_count_processed": proc_chars,
            "char_count_change": proc_chars - orig_chars,
            "sentence_count_original": orig_sentences,
            "sentence_count_processed": proc_sentences,
            "sentence_count_change": proc_sentences - orig_sentences
        }
    except Exception as e:
        print(f"Error in evaluation: {e}")
        return None

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

def paraphrase_single_text(input_text):
    """
    Paraphrase a single text using the fixed grammar and style models.
    Models are loaded on-demand.
    """
    try:
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
        
        # Run garbage collection to free memory
        gc.collect()
        
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