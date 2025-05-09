# import streamlit as st
# import os, sys
# import time
# from datetime import datetime
# from zoneinfo import ZoneInfo
# import pandas as pd
# from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
# from tqdm import tqdm
# import gc  # For garbage collection
# import time

# from transformers import pipeline
# from tqdm import tqdm

# GRAMMAR_MODELS = [
#     "google/flan-t5-small",
#     "facebook/bart-base"
# ]
# STYLE_MODELS = [
#     "google/flan-t5-base",
#     "sshleifer/distilbart-cnn-12-6"
# ]

# # Using CPU
# DEVICE = "cpu"
# print("Using CPU for processing")
    
# BATCH_SIZE = 1 
# MAX_LENGTH = 512

# grammar_pipes = {
#     m: pipeline(
#         "text2text-generation",
#         model=m,
#         device=DEVICE,
#         do_sample=False,
#         num_beams=4,
#         max_length=MAX_LENGTH
#     )
#     for m in GRAMMAR_MODELS
# }

# style_pipes = {
#     m: pipeline(
#         "text2text-generation",
#         model=m,
#         device=DEVICE,
#         do_sample=True,
#         temperature=0.9,
#         top_p=0.8,
#         num_beams=4,
#         max_length=MAX_LENGTH / 2
#     )
#     for m in STYLE_MODELS
# }


# def preprocess_texts(texts):
#     return [" ".join(t.strip().split()) for t in texts]

# def run_pipe_with_prefix(
#     pipe,
#     texts: list[str],
#     model_name: str,
#     is_grammar: bool,
#     batch_size: int = BATCH_SIZE,
#     max_len: int = MAX_LENGTH,
# ):
    
#     # Determine model family → choose prefixes
#     t5 = "t5" in model_name.lower()
#     if t5:
#         gram_pref  = "Fix all grammar and spelling errors> "
#         style_pref = "Paraphase and Simplify> "
#     else:
#         gram_pref  = "Fix all grammar and spelling errors> "
#         style_pref = "Paraphase and Simplify> "

#     prefix = gram_pref if is_grammar else style_pref
#     strip_prefixes = (gram_pref, style_pref, "Paraphrase: ", "Grammar correction: ")

#     outputs_all = []
    
#     for i in tqdm(
#         range(0, len(texts), batch_size),
#         desc=f"{'Grammar' if is_grammar else 'Style'} pass ({model_name})",
#         leave=False,
#     ):
#         chunk = texts[i : i + batch_size]

#         # 1) add prefix
#         inp_with_pref = [prefix + txt for txt in chunk]

#         # 2) pipeline call
#         outs = pipe(
#             inp_with_pref,
#             max_length=max_len,
#             batch_size=batch_size,
#         )

#         # 3) decode & strip prefixes
#         for out in outs:
#             decoded = out["generated_text"].strip()
#             for p in strip_prefixes:
#                 if decoded.startswith(p):
#                     decoded = decoded[len(p):].lstrip()
#             outputs_all.append(decoded)

#     return outputs_all


# class UploadRecord:
#     def __init__(self):
#         # self.upload_dir = "strm/uploads"
#         self.upload_dir = os.path.join("Aditya", "strm", "uploads")
#         os.makedirs(self.upload_dir, exist_ok=True)

#         if 'file_already_saved' not in st.session_state:
#             st.session_state.file_already_saved = False
#         if 'saved_file_path' not in st.session_state:
#             st.session_state.saved_file_path = None
#         if 'upload_transcription_result' not in st.session_state:
#             st.session_state.upload_transcription_result  = None
        
        
        

#     def save_uploaded_file(self, uploaded_file):
#         # Prevent saving the same file multiple times
#         if st.session_state.file_already_saved and st.session_state.saved_file_path:
#             return st.session_state.saved_file_path
        
#         try:
#             if uploaded_file is None:
#                 return None
            
#             file_timestamp = datetime.now(ZoneInfo("Australia/Sydney")).strftime("%Y-%m-%d_%H%M%S")
#             print(f"file_timestamp: {file_timestamp}")
#             file_name, file_extension = os.path.splitext(uploaded_file.name)
#             f_file_name = f"{file_name}_{file_timestamp}{file_extension}"
#             print(f"f_file_name: {f_file_name}")

#             file_path = os.path.join(self.upload_dir, f_file_name)
#             with open(file_path, "wb") as f:
#                 f.write(uploaded_file.getbuffer())

#             st.session_state.file_already_saved = True
#             st.session_state.saved_file_path = file_path
#             return file_path

#         except Exception as e:
#             print(f"Error: {e}")
#             return False


#     def transcribe(self):
#         st.title("Audio File Upload")
#         st.write("Please upload your recording")

#         # Reset file_already_saved when a new file is uploaded
#         uploaded_file = st.file_uploader(
#             label="Recording",
#             type=["wav", "mp3"],
#             accept_multiple_files=False,
#             key="file_uploader"
#         )
        
#         # If the uploader is empty, reset the session state
#         if uploaded_file is None:
#             st.session_state.file_already_saved = False
#             st.session_state.saved_file_path = None
#             st.session_state.upload_transcription_result = None
        
#         print(f"uploaded_file received: {uploaded_file}")

#         if uploaded_file is not None:
#             file_path = self.save_uploaded_file(uploaded_file)
#             if file_path is None or file_path is False:
#                 st.write("Something went wrong saving the file. Please retry.")
#             else:
#                 st.html("<h1 style='color: green;'>File uploaded successfully.</h1>")
                
                
#                 transcribe_button = st.button(
#                     label="Transcribe 📝", 
#                     disabled=(file_path is None),
#                     key="transcribe_button"
#                 )
                
#                 if transcribe_button:
#                     with st.spinner("Transcribing audio..."):
#                         try:
#                             from oai_whisper import Transcriber
                            
#                             transcriber = Transcriber()
#                             result = transcriber.transcribe_audio(file_path)
                            
#                             if result:
#                                 st.session_state.upload_transcription_result = result
#                             else:
#                                 st.error("Transcription failed. Please try again.")
#                         except Exception as e:
#                             st.error(f"Error during transcription: {str(e)}")
#                             print(f"Transcription error: {str(e)}")
                

#                 if st.session_state.upload_transcription_result:
#                     st.success("Transcription complete!\nYou can edit the text below:")
#                     edited_transcription = st.text_area(
#                         "Edit Transcription",
#                         value=st.session_state.upload_transcription_result,
#                         height=300,
#                         key="upload_editable_transcription"
#                     )
                    

#                     save_trans_button = st.button(
#                         "Save Transcription", 
#                         key="upload_save_trans_button"
#                     )
                    
#                     if save_trans_button:
#                         print("Saving edited transcription...")
#                         transcript_filename = os.path.basename(file_path).split('.')[0] + "_transcript.txt"
#                         transcript_path = os.path.join(self.upload_dir, transcript_filename)
                        
#                         with open(transcript_path, "w", encoding="utf-8") as f:
#                             f.write(edited_transcription)
                        
#                         print(f"Transcription saved to {transcript_path}") 
#                         st.success(f"Transcription saved successfully.")

#                         # -----------  PARAPHRASING ACTION/ASSESSMENT  -----------
#                         time.sleep(0.2)
#                         with st.spinner("🔍 Analyzing your text..."):
#                             st.caption("Processing your text to improve clarity and presentation")
                                
#                             paraphrased_text = paraphrase_single_text(
#                                 edited_transcription,
#                                 "google/flan-t5-small",  # Grammar model
#                                 "sshleifer/distilbart-cnn-12-6"    # Style model
#                             )
                            

#                             st.success("Analysis complete! ✨")
#                             st.write(paraphrased_text)


# def paraphrase_single_text(input_text, grammar_model_name, style_model_name):
#     """
#     Paraphrase a single text using specified grammar and style models.
    
#     Args:
#         input_text (str): The text to paraphrase
#         grammar_model_name (str): Name of the model for grammar correction
#         style_model_name (str): Name of the model for styling/paraphrasing
        
#     Returns:
#         str: The paraphrased text
#     """
#     import torch
#     from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
#     import gc
    
#     try:

#         clean_texts = preprocess_texts([input_text])
#         corrected_cache = {}

#         g_pipe  = grammar_pipes[grammar_model_name]
#         g_fixed = run_pipe_with_prefix(
#             g_pipe, clean_texts, grammar_model_name, is_grammar=True
#         )
#         g_clean = preprocess_texts(g_fixed)

#         s_pipe = style_pipes[style_model_name]
#         preds  = run_pipe_with_prefix(
#             s_pipe, g_clean, style_model_name, is_grammar=False
#         )

#         return preds[0]
        
#     except Exception as e:
#         print(f"Error during paraphrasing: {e}")
#         import traceback
#         traceback.print_exc()
#         return f"Error during paraphrasing: {str(e)}"
    

# def show_upload_page():
#     """Function that encapsulates the upload page functionality"""
#     upr = UploadRecord()
#     upr.transcribe()

# if __name__ == "__main__":
#     show_upload_page()



# /////////////////////////////////////////////////////////////////////////////////////////////////////////
import streamlit as st
import os, sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd
import numpy as np
from tqdm import tqdm
import gc  # For garbage collection

# Dictionary to store loaded models (only load when needed)
grammar_pipes = {}
style_pipes = {}

GRAMMAR_MODELS = [
    "google/flan-t5-small",
    "facebook/bart-base"
]
STYLE_MODELS = [
    "sshleifer/distilbart-cnn-12-6",
    "rajistics/informal_formal_style_transfer"  # Added from the notebook
]

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
    return [" ".join(t.strip().split()) for t in texts]

def run_pipe_with_prefix(
    pipe,
    texts: list[str],
    model_name: str,
    is_grammar: bool,
    batch_size: int = BATCH_SIZE,
    max_len: int = MAX_LENGTH,
):
    
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
    Only loads evaluation libraries when called
    """
    try:
        # Import evaluation libraries here to avoid loading them at startup
        import textstat
        
        # Calculate readability scores
        fkgl_orig = textstat.flesch_kincaid_grade(original_text)
        fkgl_proc = textstat.flesch_kincaid_grade(processed_text)
        fre_orig = textstat.flesch_reading_ease(original_text)
        fre_proc = textstat.flesch_reading_ease(processed_text)
        
        # Skip perplexity (requires heavy models) unless specifically requested
        # We could add a parameter for this if needed
        
        return {
            "fkgl_original": fkgl_orig,
            "fkgl_processed": fkgl_proc,
            "fkgl_change": fkgl_orig - fkgl_proc,
            "fre_original": fre_orig,
            "fre_processed": fre_proc,
            "fre_change": fre_proc - fre_orig
        }
    except ImportError:
        print("textstat not available for evaluation")
        return None
    except Exception as e:
        print(f"Error in evaluation: {e}")
        return None

def paraphrase_single_text(input_text, grammar_model_name, style_model_name):
    """
    Paraphrase a single text using specified grammar and style models.
    Models are loaded on-demand.
    """
    try:
        # Only load the transformers import when needed
        from transformers import pipeline
        
        clean_texts = preprocess_texts([input_text])
        
        # Load and run grammar model
        g_pipe = load_model_if_needed(grammar_model_name, is_grammar=True)
        g_fixed = run_pipe_with_prefix(
            g_pipe, clean_texts, grammar_model_name, is_grammar=True
        )
        g_clean = preprocess_texts(g_fixed)
        
        # Load and run style model
        s_pipe = load_model_if_needed(style_model_name, is_grammar=False)
        preds = run_pipe_with_prefix(
            s_pipe, g_clean, style_model_name, is_grammar=False
        )
        
        # Run garbage collection to free memory
        gc.collect()
        
        return preds[0], g_fixed[0]  # Return both the final result and grammar-corrected text
        
    except Exception as e:
        print(f"Error during paraphrasing: {e}")
        import traceback
        traceback.print_exc()
        return f"Error during paraphrasing: {str(e)}", input_text


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

    print("\n--- MarianMT (en→fr→en) Paraphrasing ---")
    # English to French
    en_fr_model_name = "Helsinki-NLP/opus-mt-en-fr"
    fr_en_model_name = "Helsinki-NLP/opus-mt-fr-en"
    en_fr_tokenizer = MarianTokenizer.from_pretrained(en_fr_model_name)
    en_fr_model = MarianMTModel.from_pretrained(en_fr_model_name)
    fr_en_tokenizer = MarianTokenizer.from_pretrained(fr_en_model_name)
    fr_en_model = MarianMTModel.from_pretrained(fr_en_model_name)

    # English to French
    fr_tokens = en_fr_tokenizer([bad_text], return_tensors="pt", padding=True)
    fr_translation = en_fr_model.generate(**fr_tokens)
    fr_text = en_fr_tokenizer.batch_decode(fr_translation, skip_special_tokens=True)[0]

    # French back to English
    en_tokens = fr_en_tokenizer([fr_text], return_tensors="pt", padding=True)
    en_translation = fr_en_model.generate(**en_tokens)
    marian_result = fr_en_tokenizer.batch_decode(en_translation, skip_special_tokens=True)[0]
    print(marian_result)
    return marian_result

    # ############################
    # # mBART Round-trip Paraphrasing (en->de->en)
    # ############################
    # print("\n--- mBART (en→de→en) Paraphrasing ---")
    # mbart_model_name = "facebook/mbart-large-50-many-to-many-mmt"
    # mbart_tokenizer = MBart50TokenizerFast.from_pretrained(mbart_model_name)
    # mbart_model = MBartForConditionalGeneration.from_pretrained(mbart_model_name)

    # # English to German
    # mbart_tokenizer.src_lang = "en_XX"
    # en_tokens = mbart_tokenizer(bad_text, return_tensors="pt")
    # de_ids = mbart_model.generate(**en_tokens, forced_bos_token_id=mbart_tokenizer.lang_code_to_id["de_DE"])
    # de_text = mbart_tokenizer.decode(de_ids[0], skip_special_tokens=True)

    # # German back to English
    # mbart_tokenizer.src_lang = "de_DE"
    # de_tokens = mbart_tokenizer(de_text, return_tensors="pt")
    # en_ids = mbart_model.generate(**de_tokens, forced_bos_token_id=mbart_tokenizer.lang_code_to_id["en_XX"])
    # mbart_result = mbart_tokenizer.decode(en_ids[0], skip_special_tokens=True)
    # print(mbart_result)
    # return mbart_result

# Input: Replace with your own bad text


class UploadRecord:
    def __init__(self):
        # self.upload_dir = "strm/uploads"
        self.upload_dir = os.path.join("Aditya", "strm", "uploads")
        os.makedirs(self.upload_dir, exist_ok=True)

        if 'file_already_saved' not in st.session_state:
            st.session_state.file_already_saved = False
        if 'saved_file_path' not in st.session_state:
            st.session_state.saved_file_path = None
        if 'upload_transcription_result' not in st.session_state:
            st.session_state.upload_transcription_result = None
        if 'paraphrased_text' not in st.session_state:
            st.session_state.paraphrased_text = None
        if 'grammar_corrected_text' not in st.session_state:
            st.session_state.grammar_corrected_text = None
        

    def save_uploaded_file(self, uploaded_file):
        # Prevent saving the same file multiple times
        if st.session_state.file_already_saved and st.session_state.saved_file_path:
            return st.session_state.saved_file_path
        
        try:
            if uploaded_file is None:
                return None
            
            file_timestamp = datetime.now(ZoneInfo("Australia/Sydney")).strftime("%Y-%m-%d_%H%M%S")
            print(f"file_timestamp: {file_timestamp}")
            file_name, file_extension = os.path.splitext(uploaded_file.name)
            f_file_name = f"{file_name}_{file_timestamp}{file_extension}"
            print(f"f_file_name: {f_file_name}")

            file_path = os.path.join(self.upload_dir, f_file_name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.session_state.file_already_saved = True
            st.session_state.saved_file_path = file_path
            return file_path

        except Exception as e:
            print(f"Error: {e}")
            return False


    def transcribe(self):
        st.title("Audio File Upload")
        st.write("Please upload your recording")

        # Reset file_already_saved when a new file is uploaded
        uploaded_file = st.file_uploader(
            label="Recording",
            type=["wav", "mp3"],
            accept_multiple_files=False,
            key="file_uploader"
        )
        
        # If the uploader is empty, reset the session state
        if uploaded_file is None:
            st.session_state.file_already_saved = False
            st.session_state.saved_file_path = None
            st.session_state.upload_transcription_result = None
            st.session_state.paraphrased_text = None
            st.session_state.grammar_corrected_text = None
        
        print(f"uploaded_file received: {uploaded_file}")

        if uploaded_file is not None:
            file_path = self.save_uploaded_file(uploaded_file)
            if file_path is None or file_path is False:
                st.write("Something went wrong saving the file. Please retry.")
            else:
                st.success("File uploaded successfully")
                
                transcribe_button = st.button(
                    label="Transcribe 📝", 
                    disabled=(file_path is None),
                    key="transcribe_button"
                )
                
                if transcribe_button:
                    with st.spinner("Transcribing audio..."):
                        try:
                            from oai_whisper import Transcriber
                            
                            transcriber = Transcriber()
                            result = transcriber.transcribe_audio(file_path)
                            
                            if result:
                                st.session_state.upload_transcription_result = result
                            else:
                                st.error("Transcription failed. Please try again.")
                        except Exception as e:
                            st.error(f"Error during transcription: {str(e)}")
                            print(f"Transcription error: {str(e)}")
                

                if st.session_state.upload_transcription_result:
                    st.success("Transcription complete! You can edit the text below:")
                    edited_transcription = st.text_area(
                        "Edit Transcription",
                        value=st.session_state.upload_transcription_result,
                        height=300,
                        key="upload_editable_transcription"
                    )
                    
                    # Add a section for model selection
                    st.subheader("Text Processing Options")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        grammar_model = st.selectbox(
                            "Grammar Model",
                            options=GRAMMAR_MODELS,
                            format_func=lambda x: x.split('/')[-1],
                            index=0
                        )
                    
                    with col2:
                        style_model = st.selectbox(
                            "Style/Simplification Model",
                            options=STYLE_MODELS,
                            format_func=lambda x: x.split('/')[-1],
                            index=0
                        )

                    save_trans_button = st.button(
                        "Process Text", 
                        key="upload_save_trans_button"
                    )
                    
                    if save_trans_button:
                        print("Saving edited transcription...")
                        transcript_filename = os.path.basename(file_path).split('.')[0] + "_transcript.txt"
                        transcript_path = os.path.join(self.upload_dir, transcript_filename)
                        
                        with open(transcript_path, "w", encoding="utf-8") as f:
                            f.write(edited_transcription)
                        
                        print(f"Transcription saved to {transcript_path}") 
                        st.success(f"Transcription saved successfully.")

                        # -----------  PARAPHRASING ACTION/ASSESSMENT  -----------
                        time.sleep(0.2)
                        with st.spinner("🔍 Processing your text..."):
                            st.caption("Improving clarity and presentation...")
                                
                            paraphrased_text, grammar_corrected_text = paraphrase_single_text(
                                edited_transcription,
                                grammar_model,  # Selected Grammar model
                                style_model     # Selected Style model
                            )
                            
                            st.session_state.paraphrased_text = paraphrased_text
                            st.session_state.grammar_corrected_text = grammar_corrected_text
                            
                            # Calculate evaluation metrics
                            metrics = evaluate_text_quality(edited_transcription, paraphrased_text)



                    # Display results if available
                    if st.session_state.paraphrased_text:
                        st.success("Processing complete! ✨")
                        
                        # Show the processed text
                        st.subheader("Processed Text Result")
                        # st.write(st.session_state.paraphrased_text)
                        st.html(f"<h2 style='color: white'; font-weight: bold>{st.session_state.paraphrased_text}</h2>")
                        translate_model_text = test_with_translation_models(edited_transcription)
                        if translate_model_text:
                            st.write("Translation model response:")
                            st.html(f"<h2 style='color: yellow;font-weight: italic'>{translate_model_text}</h2>")
                        
                        # Add download button for processed text
                        processed_filename = os.path.basename(file_path).split('.')[0] + "_processed.txt"
                        st.download_button(
                            label="Download Processed Text",
                            data=st.session_state.paraphrased_text,
                            file_name=processed_filename,
                            mime="text/plain"
                        )
                        
                        # Show the intermediate grammar correction
                        with st.expander("View intermediate grammar correction"):
                            st.write(st.session_state.grammar_corrected_text)
                        
                        # Show metrics if available
                        if 'metrics' in locals() and metrics:
                            with st.expander("Text Quality Metrics"):
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.metric(
                                        "Readability Grade Level", 
                                        f"{metrics['fkgl_processed']:.1f}",
                                        f"{-metrics['fkgl_change']:.1f}"
                                    )
                                
                                with col2:
                                    st.metric(
                                        "Reading Ease", 
                                        f"{metrics['fre_processed']:.1f}",
                                        f"{metrics['fre_change']:.1f}"
                                    )
                                
                                st.caption("Lower grade level = easier to read, Higher reading ease = better")


def show_upload_page():
    """Function that encapsulates the upload page functionality"""
    upr = UploadRecord()
    upr.transcribe()

if __name__ == "__main__":
    show_upload_page()