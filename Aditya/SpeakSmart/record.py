import streamlit as st
import os
import time
from datetime import datetime
from zoneinfo import ZoneInfo
from Ollama_client import chat_with_models
import re
import utility

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


class AudioRecorder:
    def __init__(self):
        self.upload_dir = os.path.join("Aditya", "strm", "uploads")
        os.makedirs(self.upload_dir, exist_ok=True)
        
        if 'recorded_audio' not in st.session_state:
            st.session_state.recorded_audio = None
        if 'record_transcription_result' not in st.session_state:
            st.session_state.record_transcription_result = None
        if 'paraphrased_text' not in st.session_state:
            st.session_state.paraphrased_text = None
        if 'grammar_corrected_text' not in st.session_state:
            st.session_state.grammar_corrected_text = None

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
                st.success("Transcription complete! You can edit the text below:")
                edited_transcription = st.text_area(
                    "Edit Transcription",
                    value=st.session_state.record_transcription_result,
                    height=300,
                    key="record_editable_transcription"
                )
                
                # Add info about the models being used
                # st.info(f"Using {utility.GRAMMAR_MODEL.split('/')[-1]} for grammar correction and {utility.STYLE_MODEL.split('/')[-1]} for text refinement")

                if st.button("Process Text", key="record_save_trans_button"):
                    transcript_filename = f"recording_{file_timestamp}_transcript.txt"
                    transcript_path = os.path.join(self.upload_dir, transcript_filename)
                    
                    # Use utility function to save text
                    utility.save_text_to_file(edited_transcription, transcript_path)
                    st.success(f"Transcription saved successfully.")
                    
                    # -----------  PARAPHRASING ACTION/ASSESSMENT  -----------
                    time.sleep(0.2)
                    with st.spinner("🔍 Processing your text..."):
                        st.caption("Improving clarity and presentation...")
                            
                        paraphrased_text, grammar_corrected_text = utility.paraphrase_single_text(
                            edited_transcription
                        )
                        
                        st.session_state.paraphrased_text = paraphrased_text
                        st.session_state.grammar_corrected_text = grammar_corrected_text
                
                # Display results if available
                if 'paraphrased_text' in st.session_state and st.session_state.paraphrased_text:
                    st.success("Processing complete! ✨")
                    
                    # Show the processed text
                    st.subheader("Processed Text Result")

                    st.html(f"<h3>Using {utility.GRAMMAR_MODEL.split('/')[-1]} for grammar correction and {utility.STYLE_MODEL.split('/')[-1]} for text refinement</h3>")
                    st.write(st.session_state.paraphrased_text)
                    
                    # # Add download button for processed text
                    # processed_filename = f"recording_{file_timestamp}_processed.txt"
                    # st.download_button(
                    #     label="Download Processed Text",
                    #     data=st.session_state.paraphrased_text,
                    #     file_name=processed_filename,
                    #     mime="text/plain"
                    # )
                    
                    # Show the intermediate grammar correction
                    with st.expander("View intermediate grammar correction"):
                        st.write(st.session_state.grammar_corrected_text)
                    
                    # Use our utility method to display metrics
                    utility.display_text_quality_metrics(
                        edited_transcription, 
                        st.session_state.paraphrased_text,
                        title="Text Quality Metrics"
                    )

def show_record_page():
    """Function that encapsulates the record page functionality"""
    recorder = AudioRecorder()
    recorder.record_audio()

# This allows the file to work both as a standalone and as an imported module
if __name__ == "__main__":
    show_record_page()