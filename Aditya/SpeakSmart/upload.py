import streamlit as st
import os
import time
from datetime import datetime
from zoneinfo import ZoneInfo
from Ollama_client import chat_with_models
import re
import utility
import torch

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


class UploadRecord:
    def __init__(self):
        self.upload_dir = os.path.join(os.getcwd(), "uploads")
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
                    
                    # Add option to choose processing method
                    processing_method = st.radio(
                        "Choose processing method:",
                        ["Parallel Processing (Faster)", "Sequential Processing (Original)"],
                        key="processing_method"
                    )
                    
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

                        # Create a placeholder for results
                        results_placeholder = st.container()
                        
                        with results_placeholder:
                            # Choose processing method
                            if processing_method == "Parallel Processing (Faster)":
                                # Use parallel processing
                                st.info("🚀 Running all models in parallel...")
                                
                                # Create a progress bar
                                progress_bar = st.progress(0)
                                status_text = st.empty()
                                
                                # Start parallel processing
                                start_time = time.time()
                                with st.spinner("⚡ Processing with all models simultaneously..."):
                                    # Process text with all models in parallel
                                    results = utility.process_text_parallel(
                                        edited_transcription,
                                        progress_callback=lambda p: progress_bar.progress(p/100)
                                    )
                                    
                                    end_time = time.time()
                                    processing_time = end_time - start_time
                                    status_text.success(f"✨ All models completed in {processing_time:.2f} seconds!")
                                
                                # Store results in session state
                                if results["grammar_corrected"]:
                                    st.session_state.grammar_corrected_text = results["grammar_corrected"]
                                if results["style_improved"]:
                                    st.session_state.paraphrased_text = results["style_improved"]
                                
                                # Display all results
                                self.display_all_results(
                                    edited_transcription, 
                                    results, 
                                    file_path
                                )
                                
                            else:
                                # Use sequential processing (original method)
                                st.info("🐢 Running models sequentially...")
                                
                                # -----------  PARAPHRASING ACTION/ASSESSMENT  -----------
                                with st.spinner("🔍 Processing your text..."):
                                    st.caption("Improving clarity and presentation...")
                                        
                                    paraphrased_text, grammar_corrected_text = utility.paraphrase_single_text(
                                        edited_transcription
                                    )
                                    
                                    st.session_state.paraphrased_text = paraphrased_text
                                    st.session_state.grammar_corrected_text = grammar_corrected_text
                                
                                # Display initial results
                                self.display_initial_results(
                                    edited_transcription, 
                                    paraphrased_text, 
                                    grammar_corrected_text, 
                                    file_path
                                )
                                
                                # Continue with other models
                                self.continue_sequential_processing(edited_transcription)

    def display_all_results(self, edited_transcription, results, file_path):
        """Display all results from parallel processing"""
        
        # Display any errors
        if results["errors"]:
            st.error("Some errors occurred during processing:")
            for error in results["errors"]:
                st.write(f"- {error}")
        
        # Display main results
        if results["style_improved"]:
            st.success("✨ Processing complete! All models have finished.")
            
            st.markdown("""<h2 style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen,Ubuntu, 
                Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;color: #d6e3e9;
                text-shadow:2px 2px 8px #553c9a,0 0 20px #ee4b2b;">Processed Text Result</h2>""",unsafe_allow_html=True)
            
            # Main result
            st.subheader("Grammar & Style Corrected:")
            st.html(f"<h3>Using {utility.GRAMMAR_MODEL.split('/')[-1]} for grammar correction and {utility.STYLE_MODEL.split('/')[-1]} for text refinement</h3>")
            st.markdown(
                f"""
                <div style="border: 2px solid #4682b4; border-radius: 10px; padding: 15px; background-color: #f0f8ff;">
                    <p style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif; color: #333;">
                        {results["style_improved"]}
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Download button
            processed_filename = os.path.basename(file_path).split('.')[0] + "_processed.txt"
            st.download_button(
                label="Download Processed Text",
                data=results["style_improved"],
                file_name=processed_filename,
                mime="text/plain"
            )
            
            # Show intermediate result
            if results["grammar_corrected"]:
                with st.expander("View intermediate grammar correction"):
                    st.write(results["grammar_corrected"])
            
            # Display metrics
            utility.display_text_quality_metrics(
                edited_transcription, 
                results["style_improved"], 
                title="Text Quality Metrics"
            )
        
        # Display Translation results
        if results["translation"]:
            st.markdown("---")
            st.markdown("""<h3 style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen,Ubuntu, 
                Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;color: #4682b4;
                text-shadow:2px 2px 8px #1e90ff,0 0 20px #00b7eb;">Translation Model Result</h3>""",unsafe_allow_html=True)
            
            st.markdown(
                f"""
                <div style="border: 2px solid #4682b4; border-radius: 10px; padding: 15px; background-color: #f0f8ff;">
                    <p style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif; color: #333;">
                        {results["translation"]}
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
            utility.display_text_quality_metrics(
                edited_transcription, 
                results["translation"], 
                title="Text Quality Metrics: Translation Model"
            )
        
        # Display Ollama results
        if results["ollama_results"]:
            st.markdown("---")
            # Extract Ollama results
            ollama_extracted = utility.extract_from_model_responses(results["ollama_results"])
            
            # Display Llama results
            if ollama_extracted.get('llama3.1:latest'):
                st.markdown("""<h3 style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen,Ubuntu, 
                    Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;color: #4682b4;
                    text-shadow:2px 2px 8px #1e90ff,0 0 20px #00b7eb;">Llama Result</h3>""",unsafe_allow_html=True)
                
                st.markdown(
                    f"""
                    <div style="border: 2px solid #4682b4; border-radius: 10px; padding: 15px; background-color: #f0f8ff;">
                        <p style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif; color: #333;">
                            {ollama_extracted['llama3.1:latest']}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                utility.display_text_quality_metrics(
                    edited_transcription, 
                    ollama_extracted['llama3.1:latest'], 
                    title="Text Quality Metrics: Llama"
                )
            
            # Display Mistral results
            if ollama_extracted.get('mistral:7b-instruct'):
                st.markdown("""<h3 style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen,Ubuntu, 
                    Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;color: #4682b4;
                    text-shadow:2px 2px 8px #1e90ff,0 0 20px #00b7eb;">Mistral Result</h3>""",unsafe_allow_html=True)
                
                st.markdown(
                    f"""
                    <div style="border: 2px solid #4682b4; border-radius: 10px; padding: 15px; background-color: #f0f8ff;">
                        <p style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif; color: #333;">
                            {ollama_extracted['mistral:7b-instruct']}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                utility.display_text_quality_metrics(
                    edited_transcription, 
                    ollama_extracted['mistral:7b-instruct'], 
                    title="Text Quality Metrics: Mistral"
                )

    def display_initial_results(self, edited_transcription, paraphrased_text, grammar_corrected_text, file_path):
        """Display initial results for sequential processing"""
        if paraphrased_text:
            st.success("✨ Grammar and Style Models Complete!")
            
            st.markdown("""<h2 style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen,Ubuntu, 
                Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;color: #d6e3e9;
                text-shadow:2px 2px 8px #553c9a,0 0 20px #ee4b2b;">Processed Text Result</h2>""",unsafe_allow_html=True)
            st.html(f"<h3>Using {utility.GRAMMAR_MODEL.split('/')[-1]} for grammar correction and {utility.STYLE_MODEL.split('/')[-1]} for text refinement</h3>")
            
            st.markdown(
                f"""
                <div style="border: 2px solid #4682b4; border-radius: 10px; padding: 15px; background-color: #f0f8ff;">
                    <p style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif; color: #333;">
                        {paraphrased_text}
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Add download button for processed text
            processed_filename = os.path.basename(file_path).split('.')[0] + "_processed.txt"
            st.download_button(
                label="Download Processed Text",
                data=paraphrased_text,
                file_name=processed_filename,
                mime="text/plain"
            )
            
            # Show the intermediate grammar correction
            with st.expander("View intermediate grammar correction"):
                st.write(grammar_corrected_text)
            
            # Use our utility method to display metrics
            utility.display_text_quality_metrics(
                edited_transcription, 
                paraphrased_text, 
                title="Text Quality Metrics"
            )

    def continue_sequential_processing(self, edited_transcription):
        """Continue with remaining models in sequential processing"""
        st.markdown("---")
        st.markdown("### Processing additional models...")
        
        # Translation Model
        with st.spinner("🔄 Processing with Translation Model..."):
            time.sleep(0.5)  # Small delay for visual effect
            translate_model_text = utility.test_with_translation_models(edited_transcription)
        
        if translate_model_text:
            st.success("✨ Translation Model Complete!")
            st.markdown("""<h3 style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen,Ubuntu, 
            Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;color: #4682b4;
            text-shadow:2px 2px 8px #1e90ff,0 0 20px #00b7eb;">Translation model</h3>""",unsafe_allow_html=True)
            
            st.markdown(
                f"""
                <div style="border: 2px solid #4682b4; border-radius: 10px; padding: 15px; background-color: #f0f8ff;">
                    <p style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif; color: #333;">
                        {translate_model_text}
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
            utility.display_text_quality_metrics(
                edited_transcription, 
                translate_model_text, 
                title="Text Quality Metrics: Translation Model"
            )
        else:
            st.warning("Translation model did not generate a result.")
        
        # Ollama Models
        st.markdown("---")
        with st.spinner("🤖 Processing with Ollama Models (Llama & Mistral)..."):
            time.sleep(0.5)  # Small delay for visual effect
            
            try:
                base_prompt = f"Please fix grammatical errors in this sentence and improve its style: {edited_transcription}. Add it between `<fixg>` and `</fixg>` tags."
                data = chat_with_models(base_prompt, ["mistral:7b-instruct", "llama3.1:latest"])
                
                st.success("✨ Ollama Models Complete!")
                
                # Extract and display results
                llama_response = utility.extract_from_model_responses(data)['llama3.1:latest']
                mistral_response = utility.extract_from_model_responses(data)['mistral:7b-instruct']
                
                # Display Llama results
                st.markdown("""<h3 style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen,Ubuntu, 
                    Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;color: #4682b4;
                    text-shadow:2px 2px 8px #1e90ff,0 0 20px #00b7eb;">Llama</h3>""",unsafe_allow_html=True)
                
                if llama_response:
                    st.markdown(
                        f"""
                        <div style="border: 2px solid #4682b4; border-radius: 10px; padding: 15px; background-color: #f0f8ff;">
                            <p style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif; color: #333;">
                                {llama_response}
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    utility.display_text_quality_metrics(
                        edited_transcription, 
                        llama_response, 
                        title="Text Quality Metrics: Llama"
                    )
                else:
                    st.html(f"<h2 style='color: yellow;font-weight: italic'> - </h2>")

                # Display Mistral results
                st.markdown("""<h3 style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen,Ubuntu, 
                    Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;color: #4682b4;
                    text-shadow:2px 2px 8px #1e90ff,0 0 20px #00b7eb;">Mistral</h3>""",unsafe_allow_html=True)
                
                if mistral_response:
                    st.markdown(
                        f"""
                        <div style="border: 2px solid #4682b4; border-radius: 10px; padding: 15px; background-color: #f0f8ff;">
                            <p style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif; color: #333;">
                                {mistral_response}
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    utility.display_text_quality_metrics(
                        edited_transcription, 
                        mistral_response, 
                        title="Text Quality Metrics: Mistral"
                    )
                else:
                    st.html(f"<h2 style='color: yellow;font-weight: italic'> - </h2>")
                    
            except Exception as e:
                st.error(f"Error processing with Ollama models: {e}")
                print(f"Error: {e}")

def show_upload_page():
    """Function that encapsulates the upload page functionality"""
    upr = UploadRecord()
    upr.transcribe()

if __name__ == "__main__":
    show_upload_page()