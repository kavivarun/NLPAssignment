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


class AudioRecorder:
    def __init__(self):
        self.upload_dir = os.path.join(os.getcwd(), "uploads")
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
                
                # Add option to choose processing method
                processing_method = st.radio(
                    "Choose processing method:",
                    ["Parallel Processing (Faster)", "Sequential Processing (Original)"],
                    key="record_processing_method"
                )
                
                if st.button("Process Text", key="record_save_trans_button"):
                    transcript_filename = f"recording_{file_timestamp}_transcript.txt"
                    transcript_path = os.path.join(self.upload_dir, transcript_filename)
                    
                    # Use utility function to save text
                    utility.save_text_to_file(edited_transcription, transcript_path)
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
                            self.display_all_results(edited_transcription, results)
                            
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
                                grammar_corrected_text
                            )
                            
                            # Continue with other models
                            self.continue_sequential_processing(edited_transcription)

    def display_all_results(self, edited_transcription, results):
        """Display all results from parallel processing"""
        
        # Display any errors
        if results["errors"]:
            st.error("Some errors occurred during processing:")
            for error in results["errors"]:
                st.write(f"- {error}")
        
        # Display main results
        if results["style_improved"]:
            st.success("✨ Processing complete! All models have finished.")
            
            st.subheader("Processed Text Result")
            st.html(f"<h3>Using {utility.GRAMMAR_MODEL.split('/')[-1]} for grammar correction and {utility.STYLE_MODEL.split('/')[-1]} for text refinement</h3>")
            st.write(results["style_improved"])
            
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
            st.subheader("Translation Model Results")
            st.write(results["translation"])
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
                st.subheader("Llama Results")
                st.write(ollama_extracted['llama3.1:latest'])
                utility.display_text_quality_metrics(
                    edited_transcription, 
                    ollama_extracted['llama3.1:latest'], 
                    title="Text Quality Metrics: Llama"
                )
            
            # Display Mistral results
            if ollama_extracted.get('mistral:7b-instruct'):
                st.subheader("Mistral Results")
                st.write(ollama_extracted['mistral:7b-instruct'])
                utility.display_text_quality_metrics(
                    edited_transcription, 
                    ollama_extracted['mistral:7b-instruct'], 
                    title="Text Quality Metrics: Mistral"
                )

    def display_initial_results(self, edited_transcription, paraphrased_text, grammar_corrected_text):
        """Display initial results for sequential processing"""
        if paraphrased_text:
            st.success("✨ Grammar and Style Models Complete!")
            
            st.subheader("Processed Text Result")
            st.html(f"<h3>Using {utility.GRAMMAR_MODEL.split('/')[-1]} for grammar correction and {utility.STYLE_MODEL.split('/')[-1]} for text refinement</h3>")
            st.write(paraphrased_text)
            
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
            st.subheader("Translation Model Results")
            st.write(translate_model_text)
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
                st.subheader("Llama Results")
                if llama_response:
                    st.write(llama_response)
                    utility.display_text_quality_metrics(
                        edited_transcription, 
                        llama_response, 
                        title="Text Quality Metrics: Llama"
                    )
                else:
                    st.html(f"<h2 style='color: yellow;font-weight: italic'> - </h2>")

                # Display Mistral results
                st.subheader("Mistral Results")
                if mistral_response:
                    st.write(mistral_response)
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

def show_record_page():
    """Function that encapsulates the record page functionality"""
    recorder = AudioRecorder()
    recorder.record_audio()

# This allows the file to work both as a standalone and as an imported module
if __name__ == "__main__":
    show_record_page()
