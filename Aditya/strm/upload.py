import streamlit as st
import os, sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm
import gc  # For garbage collection
import time

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
    
BATCH_SIZE = 1 
MAX_LENGTH = 512


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
            st.session_state.upload_transcription_result  = None
        
        
        

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
        
        print(f"uploaded_file received: {uploaded_file}")

        if uploaded_file is not None:
            file_path = self.save_uploaded_file(uploaded_file)
            if file_path is None or file_path is False:
                st.write("Something went wrong saving the file. Please retry.")
            else:
                st.html("<h1 style='color: green;'>File uploaded successfully.</h1>")
                
                
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
                    st.success("Transcription complete!\nYou can edit the text below:")
                    edited_transcription = st.text_area(
                        "Edit Transcription",
                        value=st.session_state.upload_transcription_result,
                        height=300,
                        key="upload_editable_transcription"
                    )
                    

                    save_trans_button = st.button(
                        "Save Transcription", 
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
                        with st.spinner("🔍 Analyzing your text..."):
                            st.caption("Processing your text to improve clarity and presentation")
                                
                            paraphrased_text = paraphrase_single_text(
                                edited_transcription,
                                "google/flan-t5-small",  # Grammar model
                                "google/flan-t5-base"    # Style model
                            )
                            

                            st.success("Analysis complete! ✨")
                            st.write(paraphrased_text)


def paraphrase_single_text(input_text, grammar_model_name, style_model_name):
    """
    Paraphrase a single text using specified grammar and style models.
    
    Args:
        input_text (str): The text to paraphrase
        grammar_model_name (str): Name of the model for grammar correction
        style_model_name (str): Name of the model for styling/paraphrasing
        
    Returns:
        str: The paraphrased text
    """
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    import gc
    

    input_text = " ".join(input_text.strip().split())
    
    try:
        # Process with grammar model
        g_tokenizer = AutoTokenizer.from_pretrained(grammar_model_name)
        g_model = AutoModelForSeq2SeqLM.from_pretrained(grammar_model_name)
        print(f"Grammar model {grammar_model_name} loaded")
        
        # Prepare input for grammar model
        is_t5_grammar = "t5" in grammar_model_name.lower()
        if is_t5_grammar:
            g_input = f"Fix all grammar and spelling errors in this text: {input_text}"
        else:
            g_input = f"Grammar correction: {input_text}"

            
        # Process with grammar model
        g_inputs = g_tokenizer(g_input, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            g_outputs = g_model.generate(
                g_inputs.input_ids,
                max_length=512,
                min_length=10,
                num_beams=4,
                do_sample=True,
                temperature=0.9,
                early_stopping=True
            )
        
        # Decode and clean grammar outputs
        grammar_output = g_tokenizer.decode(g_outputs[0], skip_special_tokens=True)
        if is_t5_grammar and (grammar_output.lower().startswith("Fix all grammar") or grammar_output.lower().startswith("Grammar correction")):
            grammar_output = grammar_output[len("Fix all grammar"):].strip()
        print(f"Grammar output: {grammar_output}")
        
        # Clean up memory
        del g_model, g_tokenizer, g_inputs, g_outputs
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Process with style model
        s_tokenizer = AutoTokenizer.from_pretrained(style_model_name)
        s_model = AutoModelForSeq2SeqLM.from_pretrained(style_model_name)
        print(f"Style model {style_model_name} loaded")
        
        # Prepare input for style model
        is_t5_style = "t5" in style_model_name.lower()
        if is_t5_style:
            s_input = f"Rewrite this text to be clear, professional, and engaging while maintaining the original meaning: {grammar_output}"
        else:
            s_input = f"Paraphrase: {grammar_output}"
            
        # Process with style model
        s_inputs = s_tokenizer(s_input, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            s_outputs = s_model.generate(
                s_inputs.input_ids,
                max_length=512,
                min_length=10,
                num_beams=4,
                do_sample=True,
                temperature=0.9,
                early_stopping=True
            )
        
        # Decode and clean style outputs
        style_output = s_tokenizer.decode(s_outputs[0], skip_special_tokens=True)
        if is_t5_style:
            prefixes = ["Rewrite this text", "Paraphrase:"]
            for prefix in prefixes:
                if style_output.lower().startswith(prefix.lower()):
                    style_output = style_output[len(prefix):].strip()
        
        # Clean up memory
        del s_model, s_tokenizer, s_inputs, s_outputs
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return style_output
        
    except Exception as e:
        print(f"Error during paraphrasing: {e}")
        import traceback
        traceback.print_exc()
        return f"Error during paraphrasing: {str(e)}"

def show_upload_page():
    """Function that encapsulates the upload page functionality"""
    upr = UploadRecord()
    upr.transcribe()

if __name__ == "__main__":
    show_upload_page()