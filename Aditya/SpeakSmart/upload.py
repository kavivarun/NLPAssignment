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


# def call_ollama(model, prompt):
#     bad_text = f"""Yesterday, I go to the store and buyed some apples. 
#     They was very tasty, but I eated them to fast. Now, I wants to get more, but store is close."""
#     base_prompt = f"Please fix grammatical errors in this sentence and improve its style: {bad_text}. Add it between `<fixg>` and `</fixg>` tags."

#     data = chat_with_models(base_prompt, ["mistral:7b-instruct", "llama3.1:latest"])



#     print(data)

#     print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
#     print(data['mistral:7b-instruct'])
#     print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
#     print(data['llama3.1:latest'])

#     print("--------------------Extracted Text--------------------")
#     print(f"Mistral: {extract_from_model_responses(data)['mistral:7b-instruct']}")
#     print(f"Llama3.1: {extract_from_model_responses(data)['llama3.1:latest']}")


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
                    
                    # Add info about the models being used
                    st.info(f"Using {utility.GRAMMAR_MODEL.split('/')[-1]} for grammar correction and {utility.STYLE_MODEL.split('/')[-1]} for text refinement")

                    save_trans_button = st.button(
                        "Process Text", 
                        key="upload_save_trans_button"
                    )
                    
                    if save_trans_button:
                        print("Saving edited transcription...")
                        transcript_filename = os.path.basename(file_path).split('.')[0] + "_transcript.txt"
                        transcript_path = os.path.join(self.upload_dir, transcript_filename)
                        
                        # Use utility function to save text
                        utility.save_text_to_file(edited_transcription, transcript_path)
                        print(f"Transcription saved to {transcript_path}") 
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
                    if st.session_state.paraphrased_text:
                        st.success("Processing complete! ✨")
                        
                        # Show the processed text
                        # st.subheader("Processed Text Result")
                        st.markdown("""<h2 style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen,Ubuntu, 
                            Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;color: #d6e3e9;
                            text-shadow:2px 2px 8px #553c9a,0 0 20px #ee4b2b;">Processed Text Result</h2>""",unsafe_allow_html=True)
                        st.write(st.session_state.paraphrased_text)
                        
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
                        
                        # Use our utility method to display metrics
                        utility.display_text_quality_metrics(
                            edited_transcription, 
                            st.session_state.paraphrased_text, 
                            title="Text Quality Metrics"
                        )


def show_upload_page():
    """Function that encapsulates the upload page functionality"""
    upr = UploadRecord()
    upr.transcribe()

if __name__ == "__main__":
    show_upload_page()