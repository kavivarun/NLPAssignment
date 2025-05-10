
import os

# Test script to verify Whisper works properly
class Transcriber:
    def __init__(self):
        self.model = None

    def load_model(self, model_size="base"):
        try:
            # Import whisper only when needed
            import whisper
            if self.model is None:
                print(f"Loading whisper model: {model_size}")
                self.model = whisper.load_model(model_size)
                print("Model loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def transcribe_audio(self, audio_path, model_size="base"):
        """Transcribe the given audio file"""
        print(f"Transcribing audio file: {audio_path}")
        print(f"Audio file exists: {os.path.exists(audio_path)}")
        
        if not os.path.exists(audio_path):
            print(f"Audio file not found: {audio_path}")
            return None
        
        print(f"Audio file size: {os.path.getsize(audio_path)} bytes")
        
        # Load the model
        if not self.load_model(model_size):
            return None
        
        try:
            # Transcribe the audio
            result = self.model.transcribe(audio_path)
            transcription_text = result["text"]
            print(f"Transcription successful: {transcription_text[:100]}...")
            return transcription_text
        except Exception as e:
            print(f"Error during transcription: {str(e)}")
            return None

# # For testing the module directly
# if __name__ == "__main__":
#     app = Transcriber()
#     result = app.transcribe_audio("sample.wav")
#     if result:
#         print(f"Transcription result: {result}")
#     else:
#         print("Transcription failed")