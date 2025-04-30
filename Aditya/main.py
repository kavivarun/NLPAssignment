import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BartTokenizer, BartForConditionalGeneration, T5Tokenizer, T5ForConditionalGeneration
import torch
import nltk
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import seaborn as sns

# Download necessary NLTK data
nltk.download('punkt', quiet=True)

# Set up device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Complex sample data with issues
sample_data = [
    "The implementation of artificial intelligence in healthcare systems has, you know, revolutionized the way doctors diagnose patients and, like, predict disease outcomes, although there are still many, many challenges that need to be addressed before we can fully rely on these systems for critical medical decisions.",
    
    "Despite the fact that climate change is widely acknowledged by scientists all over the world as one of the most pressing issues of our time, many governments have failed to take adequete action to address this issue, which may lead to catastrophic consequences for future generations if we don't act now.",
    
    "The experiment, which was conducted over a period of approximately six months with a diverse group of participants from various demographic backgrounds, yielded results that were not only statistically significant but also highly relevant to our understanding of human cognitive processes in decision-making scenarios under pressure.",
    
    "When we look at the economic impact of the pandemic, we can see that it effected different sectors in different ways, with some industries, such as technology and e-commerce, actually experiencing growth, while others, like tourism and hospitality, suffering severe losses, which has lead to a kind of uneven recovery that policymakers is struggling to address effectively.",
    
    "The literature review conducted as part of this study revealled several gaps in existing research, particularly with regards to the long-term effects of the intervention on participants from lower socioeconomic backgrounds, which suggests that further investigation is needed before any definitive conclusions can be drawn about the efficacy of the program across diverse populations."
]

# Initialize models and their functions
models = {}

# Model 1: BART Paraphrase
print("Loading BART Paraphrase model...")
try:
    bart_paraphrase_tokenizer = AutoTokenizer.from_pretrained("eugenesiow/bart-paraphrase")
    bart_paraphrase_model = AutoModelForSeq2SeqLM.from_pretrained("eugenesiow/bart-paraphrase").to(device)
    
    def bart_paraphrase_function(text, max_length=150):
        inputs = bart_paraphrase_tokenizer(text, return_tensors="pt", max_length=512, truncation=True).to(device)
        
        start_time = time.time()
        
        outputs = bart_paraphrase_model.generate(
            inputs.input_ids,
            max_length=max_length,
            num_beams=5,
            temperature=0.9,
            top_p=0.9,
            do_sample=True,
            num_return_sequences=1
        )
        
        paraphrased = bart_paraphrase_tokenizer.decode(outputs[0], skip_special_tokens=True)
        time_taken = time.time() - start_time
        
        return paraphrased, time_taken
    
    models["BART Paraphrase"] = bart_paraphrase_function
    print("BART Paraphrase model loaded successfully")
except Exception as e:
    print(f"Could not load BART Paraphrase model: {e}")

# Model 2: T5 Small
print("Loading T5 Small model...")
try:
    t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
    t5_model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)
    
    def t5_paraphrase_function(text, max_length=150):
        # Add explicit paraphrasing instruction
        input_text = f"paraphrase: {text}"
        inputs = t5_tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(device)
        
        start_time = time.time()
        
        outputs = t5_model.generate(
            inputs.input_ids,
            max_length=max_length,
            num_beams=5,
            temperature=0.9,
            top_p=0.9,
            do_sample=True,
            num_return_sequences=1
        )
        
        paraphrased = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
        time_taken = time.time() - start_time
        
        return paraphrased, time_taken
    
    models["T5 Small"] = t5_paraphrase_function
    print("T5 Small model loaded successfully")
except Exception as e:
    print(f"Could not load T5 Small model: {e}")

# Model 3: BART Base with Diversity
print("Loading BART Base model...")
try:
    bart_base_tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    bart_base_model = BartForConditionalGeneration.from_pretrained("facebook/bart-base").to(device)
    
    def bart_base_diversity_function(text, max_length=150):
        inputs = bart_base_tokenizer(text, return_tensors="pt", max_length=512, truncation=True).to(device)
        
        start_time = time.time()
        
        # Use extreme settings to force diversity
        outputs = bart_base_model.generate(
            inputs.input_ids,
            max_length=max_length,
            min_length=int(len(text.split()) * 0.8),  # Force reasonable length
            num_beams=10,
            temperature=1.5,  # Very high temperature for diversity
            top_k=50,
            top_p=0.95,
            no_repeat_ngram_size=2,  # Prevent repetition of phrases
            do_sample=True,
            num_return_sequences=1,
            repetition_penalty=2.5  # Strong penalty for repetition
        )
        
        paraphrased = bart_base_tokenizer.decode(outputs[0], skip_special_tokens=True)
        time_taken = time.time() - start_time
        
        return paraphrased, time_taken
    
    models["BART Base Diversity"] = bart_base_diversity_function
    print("BART Base model loaded successfully")
except Exception as e:
    print(f"Could not load BART Base model: {e}")

# Try to load an additional higher-quality model
print("Loading additional high-quality model...")
try:
    pegasus_tokenizer = AutoTokenizer.from_pretrained("tuner007/pegasus_paraphrase")
    pegasus_model = AutoModelForSeq2SeqLM.from_pretrained("tuner007/pegasus_paraphrase").to(device)
    
    def pegasus_paraphrase_function(text, max_length=150):
        # For pegasus, the model expects this format
        input_text = "paraphrase: " + text
        
        encoding = pegasus_tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(device)
        
        start_time = time.time()
        
        outputs = pegasus_model.generate(
            input_ids=encoding.input_ids,
            max_length=max_length,
            num_beams=8,
            temperature=1.2,
            top_k=50,
            top_p=0.95,
            early_stopping=True,
            do_sample=True
        )
        
        paraphrased = pegasus_tokenizer.decode(outputs[0], skip_special_tokens=True)
        time_taken = time.time() - start_time
        
        return paraphrased, time_taken
    
    models["PEGASUS Paraphrase"] = pegasus_paraphrase_function
    print("PEGASUS Paraphrase model loaded successfully")
except Exception as e:
    print(f"Could not load PEGASUS Paraphrase model: {e}")

# Ensure we have at least one model
if not models:
    print("No models could be loaded. Please check your installation and internet connection.")
    exit(1)

# Function to calculate difference score
def calculate_difference_score(original, paraphrased):
    words_original = set(original.lower().split())
    words_paraphrased = set(paraphrased.lower().split())
    
    if len(words_original) > 0 and len(words_paraphrased) > 0:
        # Jaccard similarity (intersection over union)
        similarity = len(words_original.intersection(words_paraphrased)) / len(words_original.union(words_paraphrased))
        difference = 1 - similarity
    else:
        difference = 0
    
    return difference

# Process samples with all models
all_results = []
print("\nProcessing samples with all models...")

for i, text in enumerate(tqdm(sample_data)):
    sample_results = {"sample_id": i+1, "original": text}
    
    for model_name, model_function in models.items():
        try:
            paraphrased, time_taken = model_function(text)
            difference_score = calculate_difference_score(text, paraphrased)
            
            sample_results[f"{model_name}_paraphrased"] = paraphrased
            sample_results[f"{model_name}_time"] = time_taken
            sample_results[f"{model_name}_difference"] = difference_score
        except Exception as e:
            print(f"Error with model {model_name} on sample {i+1}: {e}")
            sample_results[f"{model_name}_paraphrased"] = f"Error: {str(e)[:50]}..."
            sample_results[f"{model_name}_time"] = 0
            sample_results[f"{model_name}_difference"] = 0
    
    all_results.append(sample_results)

# Convert to DataFrame
results_df = pd.DataFrame(all_results)

# Save full results
results_df.to_csv("all_models_comparison.csv", index=False)
print("\nFull results saved to all_models_comparison.csv")

# Calculate summary statistics for each model
summary_data = []
for model_name in models.keys():
    avg_time = results_df[f"{model_name}_time"].mean()
    avg_difference = results_df[f"{model_name}_difference"].mean()
    
    summary_data.append({
        "Model": model_name,
        "Average Processing Time (s)": avg_time,
        "Average Difference Score": avg_difference
    })

summary_df = pd.DataFrame(summary_data)
print("\nModel Performance Summary:")
print(summary_df)
summary_df.to_csv("model_performance_summary.csv", index=False)

# Generate visualization of difference scores
plt.figure(figsize=(12, 6))
model_names = list(models.keys())
avg_differences = [results_df[f"{model}_difference"].mean() for model in model_names]

plt.bar(model_names, avg_differences)
plt.title('Average Difference Score by Model')
plt.ylabel('Difference Score (higher = more different)')
plt.ylim(0, max(avg_differences) * 1.2)

for i, v in enumerate(avg_differences):
    plt.text(i, v + 0.02, f'{v:.4f}', ha='center')

plt.tight_layout()
plt.savefig('model_difference_comparison.png')
print("Performance visualization saved to model_difference_comparison.png")

# Display side-by-side comparisons for each sample
print("\nSide-by-Side Comparison of Models:")
for i, row in enumerate(all_results):
    print(f"\nSample {row['sample_id']}:")
    print(f"Original: {row['original'][:100]}...")
    
    for model_name in models.keys():
        paraphrased = row[f"{model_name}_paraphrased"]
        difference = row[f"{model_name}_difference"]
        time_taken = row[f"{model_name}_time"]
        
        print(f"\n{model_name}:")
        print(f"Paraphrased: {paraphrased[:100]}...")
        print(f"Difference Score: {difference:.4f}")
        print(f"Processing Time: {time_taken:.4f}s")
    
    print("-" * 80)