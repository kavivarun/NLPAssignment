import json
import requests
import threading
from typing import Dict, List, Any, Optional
import torch
import time

class OllamaModelChat:
    def __init__(self, model_name: str, base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.chat_url = f"{base_url}/api/chat"
        self.generate_url = f"{base_url}/api/generate"
        self.response = None
        self.is_processing = False
        self.error = None
        
        # Check if Ollama is using GPU (this is informational only)
        self._check_ollama_status()
        
    def _check_ollama_status(self):
        """Check if Ollama server is running and log information about device usage"""
        try:
            # Try to ping the Ollama server
            response = requests.get(f"{self.base_url}/api/version", timeout=5)
            if response.status_code == 200:
                print(f"Ollama server is running at {self.base_url}")
                
                # Log local device information
                if torch.cuda.is_available():
                    print(f"Local GPU available: {torch.cuda.get_device_name(0)}")
                    print("Note: Ollama manages GPU usage independently on its server")
                else:
                    print("Local GPU not available - Ollama may still use GPU on server")
            else:
                print(f"Warning: Ollama server returned status {response.status_code}")
        except Exception as e:
            print(f"Warning: Could not connect to Ollama server: {e}")
        
    def process_prompt(self, prompt: str, system_prompt: Optional[str] = None, 
                      temperature: float = 0.7) -> None:
        """Process prompt in a way that can be called in a thread"""
        self.is_processing = True
        self.error = None
        
        try:
            # Using chat API for better conversation handling
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "stream": False  # Explicitly disable streaming
            }
            
            if system_prompt:
                payload["system"] = system_prompt
                
            # Add option to keep model loaded (improves performance for multiple requests)
            payload["options"] = {
                "num_keep": 1,  # Keep the model loaded
                "num_gpu": 1 if torch.cuda.is_available() else 0  # Suggest GPU usage if available
            }
            
            # Make the API request    
            response = requests.post(self.chat_url, json=payload)
            
            # Check for HTTP errors
            if response.status_code != 200:
                self.error = f"HTTP error {response.status_code}: {response.text}"
                return
                
            # Try to parse the first line only if there's an issue
            try:
                self.response = response.json()
                # Log if we're getting device information in the response
                if 'info' in self.response:
                    print(f"Ollama processing info: {self.response['info']}")
            except json.JSONDecodeError:
                # Fall back to parsing just the first line
                first_line = response.text.split('\n')[0]
                self.response = json.loads(first_line)
                
        except Exception as e:
            self.error = str(e)
        finally:
            self.is_processing = False

    def process_prompt_alternate(self, prompt: str, system_prompt: Optional[str] = None, 
                               temperature: float = 0.7) -> None:
        """Alternative implementation using the generate API instead of chat"""
        self.is_processing = True
        self.error = None
        
        try:
            # Try using generate API instead of chat
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "temperature": temperature,
                "stream": False,
                "options": {
                    "num_keep": 1,  # Keep the model loaded
                }
            }
            
            if system_prompt:
                payload["system"] = system_prompt
                
            response = requests.post(self.generate_url, json=payload)
            
            if response.status_code != 200:
                self.error = f"HTTP error {response.status_code}: {response.text}"
                return
                
            # Parse response
            try:
                result = response.json()
                # Format as chat response
                self.response = {
                    "message": {
                        "role": "assistant",
                        "content": result["response"]
                    },
                    "model": self.model_name
                }
                
                # Log model loading information if available
                if 'load_duration' in result:
                    print(f"Model {self.model_name} load duration: {result['load_duration']}ns")
                if 'prompt_eval_duration' in result:
                    print(f"Model {self.model_name} prompt eval duration: {result['prompt_eval_duration']}ns")
                    
            except json.JSONDecodeError:
                self.error = f"Failed to parse JSON: {response.text[:100]}..."
                
        except Exception as e:
            self.error = str(e)
        finally:
            self.is_processing = False


def chat_with_models(prompt: str, models: List[str], system_prompt: Optional[str] = None) -> Dict[str, Any]:
    """Chat with multiple models in parallel and return all responses"""
    
    # Check local GPU status before starting
    if torch.cuda.is_available():
        print(f"Local GPU available for Ollama requests: {torch.cuda.get_device_name(0)}")
    else:
        print("Local GPU not available - Ollama server may still use GPU")
    
    # Create model handlers
    model_handlers = [OllamaModelChat(model) for model in models]
    
    # Create and start threads
    threads = []
    start_time = time.time()
    
    for handler in model_handlers:
        # Use the alternate implementation which provides more performance info
        thread = threading.Thread(
            target=handler.process_prompt_alternate,
            args=(prompt, system_prompt)
        )
        thread.start()
        threads.append(thread)
    
    # Wait for all threads to complete or time out
    results = {}
    for i, thread in enumerate(threads):
        model_name = models[i]
        handler = model_handlers[i]
        
        # Wait for thread to complete (with optional timeout)
        thread.join(timeout=60)  # 60 second timeout
        
        if thread.is_alive():
            results[model_name] = {"status": "timeout", "response": None}
        elif handler.error:
            results[model_name] = {"status": "error", "error": handler.error}
        else:
            results[model_name] = {
                "status": "success", 
                "response": handler.response
            }
    
    # Log total execution time
    total_time = time.time() - start_time
    print(f"Total processing time for all models: {total_time:.2f} seconds")
    
    return results

def check_ollama_gpu_usage():
    """Check if Ollama is using GPU - this is informational only"""
    try:
        # Make a simple request to see if Ollama responds with GPU info
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            print("Ollama server is accessible")
            # Note: Ollama doesn't provide direct GPU usage info via API
            # GPU usage is managed by the Ollama server automatically
            return True
        else:
            print(f"Ollama server returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"Could not connect to Ollama: {e}")
        return False

# Add GPU memory monitoring for local models
def monitor_local_gpu():
    """Monitor local GPU usage (not Ollama's GPU usage)"""
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        gpu_allocated = torch.cuda.memory_allocated(0)
        gpu_free = gpu_memory - gpu_allocated
        
        print(f"Local GPU Status:")
        print(f"  Total memory: {gpu_memory/1024**3:.2f}GB")
        print(f"  Allocated: {gpu_allocated/1024**3:.2f}GB")
        print(f"  Free: {gpu_free/1024**3:.2f}GB")
        
        return {
            "total": gpu_memory,
            "allocated": gpu_allocated,
            "free": gpu_free
        }
    else:
        print("No local GPU available")
        return None

# Utility function to check overall system status
def check_gpu_status():
    """Check overall GPU status for both local and Ollama operations"""
    print("\n=== GPU Status Check ===")
    
    # Check local GPU
    if torch.cuda.is_available():
        print(f"Local GPU: {torch.cuda.get_device_name(0)}")
        monitor_local_gpu()
    else:
        print("Local GPU: Not available")
    
    # Check Ollama server
    print("\nChecking Ollama server...")
    check_ollama_gpu_usage()
    print("========================\n")

    