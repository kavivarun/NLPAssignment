from Ollama_client import chat_with_models

import re

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

# Example usage
def extract_from_model_responses(data):
    results = {}
    
    for model_name, result in data.items():
        if result['status'] == 'success':
            response_text = result['response']['message']['content']
            fixed_text = extract_fixed_text(response_text)
            results[model_name] = fixed_text
        else:
            results[model_name] = f"Error: {result.get('error', 'Unknown error')}"
    
    return results

# Using your example data

bad_text = f"""Yesterday, I go to the store and buyed some apples. 
They was very tasty, but I eated them to fast. Now, I wants to get more, but store is close."""
base_prompt = f"Please fix grammatical errors in this sentence and improve its style: {bad_text}. Add it between `<fixg>` and `</fixg>` tags."

data = chat_with_models(base_prompt, ["mistral:latest", "llama3.1:latest"])



print(data)

print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
print(data['mistral:latest'])
print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
print(data['llama3.1:latest'])

print("--------------------Extracted Text--------------------")
print(f"Mistral: {extract_from_model_responses(data)['mistral:latest']}")
print(f"Llama3.1: {extract_from_model_responses(data)['llama3.1:latest']}")