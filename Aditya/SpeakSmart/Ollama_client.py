import json
import requests
import threading
from typing import Dict, List, Any, Optional

class OllamaModelChat:
    def __init__(self, model_name: str, base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.chat_url = f"{base_url}/api/chat"
        self.generate_url = f"{base_url}/api/generate"
        self.response = None
        self.is_processing = False
        self.error = None
        
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
                "stream": False  # Explicitly disable streaming to avoid parsing issues
            }
            
            if system_prompt:
                payload["system"] = system_prompt
            
            # Make the API request    
            response = requests.post(self.chat_url, json=payload)
            
            # Check for HTTP errors
            if response.status_code != 200:
                self.error = f"HTTP error {response.status_code}: {response.text}"
                return
                
            # Try to parse the first line only if there's an issue
            try:
                self.response = response.json()
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
                "stream": False
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
            except json.JSONDecodeError:
                self.error = f"Failed to parse JSON: {response.text[:100]}..."
                
        except Exception as e:
            self.error = str(e)
        finally:
            self.is_processing = False


def chat_with_models(prompt: str, models: List[str], system_prompt: Optional[str] = None) -> Dict[str, Any]:
    """Chat with multiple models in parallel and return all responses"""
    
    # Create model handlers
    model_handlers = [OllamaModelChat(model) for model in models]
    
    # Create and start threads
    threads = []
    for handler in model_handlers:
        # Try the alternate implementation if you continue having issues
        thread = threading.Thread(
            target=handler.process_prompt_alternate,  # Using generate API instead
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
    
    return results

