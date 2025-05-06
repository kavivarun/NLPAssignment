import streamlit as st
import os
import time
from datetime import datetime
from zoneinfo import ZoneInfo

class AudioRecorder:
    def __init__(self):
        self.upload_dir = os.path.join("Aditya", "strm", "uploads")
        os.makedirs(self.upload_dir, exist_ok=True)
        
        if 'recorded_audio' not in st.session_state:
            st.session_state.recorded_audio = None
        if 'record_transcription_result' not in st.session_state:
            st.session_state.record_transcription_result = None

    def record_audio(self):
        st.title("Record Audio")
        st.write("Use the microphone to record your audio")
        
        # The audio_input component handles recording automatically
        recorded_audio = st.audio_input("Click to record audio")
        
        if recorded_audio is not None:
            # Display success message when audio is recorded
            st.success("Audio recorded successfully!")
            
            # Save the recorded audio
            file_timestamp = datetime.now(ZoneInfo("Australia/Sydney")).strftime("%Y-%m-%d_%H%M%S")
            file_path = os.path.join(self.upload_dir, f"recording_{file_timestamp}.wav")
            
            with open(file_path, "wb") as f:
                f.write(recorded_audio.getbuffer())
            
            st.session_state.recorded_audio = file_path
            
            # Transcribe button
            if st.button("Transcribe Recording 📝"):
                with st.spinner("Transcribing audio..."):
                    try:
                        from oai_whisper import Transcriber
                        
                        transcriber = Transcriber()
                        result = transcriber.transcribe_audio(file_path)
                        
                        if result:
                            st.session_state.record_transcription_result = result
                        else:
                            st.error("Transcription failed. Please try again.")
                    except Exception as e:
                        st.error(f"Error during transcription: {str(e)}")
                        print(f"Transcription error: {str(e)}")
            
            # Display transcription result if available
            if 'record_transcription_result' in st.session_state and st.session_state.record_transcription_result:
                st.success("Transcription complete!\nYou can edit the text below:")
                edited_transcription = st.text_area(
                    "Edit Transcription",
                    value=st.session_state.record_transcription_result,
                    height=300,
                    key="record_editable_transcription"
                )

                if st.button("Save Transcription", key="record_save_trans_button"):
                    transcript_filename = f"recording_{file_timestamp}_transcript.txt"
                    transcript_path = os.path.join(self.upload_dir, transcript_filename)
                    
                    with open(transcript_path, "w", encoding="utf-8") as f:
                        f.write(edited_transcription)
                    
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

def show_record_page():
    """Function that encapsulates the record page functionality"""
    recorder = AudioRecorder()
    recorder.record_audio()

# This allows the file to work both as a standalone and as an imported module
if __name__ == "__main__":
    show_record_page()