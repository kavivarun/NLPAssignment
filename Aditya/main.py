import torch
import pandas as pd
import numpy as np
import time
import nltk
nltk.download('punkt')
nltk.download('wordnet')

from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    pipeline,
    T5ForConditionalGeneration, 
    BartForConditionalGeneration
)
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings("ignore")

# Download necessary NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('wordnet')
except LookupError:
    nltk.download('wordnet')

class ParaphraseEvaluator:
    def __init__(self, device=None):
        """Initialize the paraphrase evaluator with device detection."""
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        
        # Initialize sentence transformer for semantic similarity
        self.sim_model = SentenceTransformer('all-MiniLM-L6-v2').to(self.device)
        
        # Initialize models dict
        self.models = {}
        self.tokenizers = {}
        
    def load_model(self, model_name, model_id):
        """Load a model and its tokenizer."""
        print(f"Loading {model_name}...")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            
            if 't5' in model_id.lower():
                model = T5ForConditionalGeneration.from_pretrained(model_id).to(self.device)
            elif 'bart' in model_id.lower():
                model = BartForConditionalGeneration.from_pretrained(model_id).to(self.device)
            else:
                model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(self.device)
                
            self.models[model_name] = model
            self.tokenizers[model_name] = tokenizer
            print(f"Successfully loaded {model_name}")
            return True
        except Exception as e:
            print(f"Error loading {model_name}: {e}")
            return False
    
    def load_dataset(self, split='test'):
        """Load the MRPC dataset."""
        print(f"Loading MRPC dataset ({split} split)...")
        dataset = load_dataset("glue", "mrpc", split=split)
        return dataset
    
    def generate_paraphrase(self, model_name, text, max_length=128, num_beams=4):
        """Generate a paraphrase using the specified model."""
        model = self.models[model_name]
        tokenizer = self.tokenizers[model_name]
        
        # Different prefix for different models
        if 't5' in model_name.lower():
            prefix = "paraphrase: "
            input_text = prefix + text
        else:
            input_text = text
            
        inputs = tokenizer(input_text, return_tensors="pt", max_length=max_length, truncation=True).to(self.device)
        
        # Track inference time
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True
            )
            
        inference_time = time.time() - start_time
        
        paraphrase = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return paraphrase, inference_time
    
    def calculate_semantic_similarity(self, text1, text2):
        """Calculate semantic similarity between two texts using sentence embeddings."""
        # Encode sentences to get embeddings
        embedding1 = self.sim_model.encode(text1, convert_to_tensor=True)
        embedding2 = self.sim_model.encode(text2, convert_to_tensor=True)
        
        # Calculate cosine similarity
        similarity = util.pytorch_cos_sim(embedding1, embedding2).item()
        return similarity
    
    def calculate_lexical_diversity(self, original, paraphrase):
        """Calculate lexical diversity metrics between original and paraphrase."""
        # Simple tokenization by splitting on whitespace instead of using nltk.word_tokenize
        original_tokens = original.lower().split()
        paraphrase_tokens = paraphrase.lower().split()
        
        # Calculate word overlap
        original_set = set(original_tokens)
        paraphrase_set = set(paraphrase_tokens)
        
        # Jaccard similarity (lower means more diverse)
        if len(original_set.union(paraphrase_set)) > 0:
            jaccard = len(original_set.intersection(paraphrase_set)) / len(original_set.union(paraphrase_set))
        else:
            jaccard = 0
            
        # Calculate a simple n-gram overlap instead of BLEU
        def get_ngrams(tokens, n=2):
            return set(' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1))
        
        original_bigrams = get_ngrams(original_tokens)
        paraphrase_bigrams = get_ngrams(paraphrase_tokens)
        
        if len(original_bigrams.union(paraphrase_bigrams)) > 0:
            bigram_overlap = len(original_bigrams.intersection(paraphrase_bigrams)) / len(original_bigrams.union(paraphrase_bigrams))
        else:
            bigram_overlap = 0
                
        return {
            "jaccard_similarity": jaccard,
            "bleu": bigram_overlap,  # Using bigram overlap as a BLEU approximation
            "diversity_score": 1 - (jaccard + bigram_overlap) / 2  # Higher means more diverse
        }
    
    def evaluate_model(self, model_name, dataset, num_samples=100):
        """Evaluate a model on the dataset."""
        if num_samples > len(dataset):
            num_samples = len(dataset)
            
        # Select a subset of the dataset
        indices = np.random.choice(len(dataset), num_samples, replace=False)
        subset = dataset.select(indices)
        
        results = []
        
        for i, item in enumerate(tqdm(subset, desc=f"Evaluating {model_name}")):
            original = item['sentence1']
            
            # Generate paraphrase
            paraphrase, inference_time = self.generate_paraphrase(model_name, original)
            
            # Calculate metrics
            semantic_sim = self.calculate_semantic_similarity(original, paraphrase)
            lexical_div = self.calculate_lexical_diversity(original, paraphrase)
            
            # Store results
            results.append({
                "original": original,
                "reference": item['sentence2'],
                "paraphrase": paraphrase,
                "semantic_similarity": semantic_sim,
                "lexical_diversity": lexical_div["diversity_score"],
                "jaccard_similarity": lexical_div["jaccard_similarity"],
                "bleu": lexical_div["bleu"],
                "inference_time": inference_time
            })
            
        return pd.DataFrame(results)
    
    def compare_models(self, model_configs, num_samples=100):
        """Compare multiple models on the dataset."""
        # Load dataset
        dataset = self.load_dataset()
        
        # Load models
        for name, model_id in model_configs.items():
            self.load_model(name, model_id)
        
        # Evaluate each model
        all_results = {}
        for model_name in model_configs.keys():
            all_results[model_name] = self.evaluate_model(model_name, dataset, num_samples)
        
        return all_results
    
    def visualize_results(self, all_results):
        """Visualize the comparison results."""
        # Prepare summary statistics
        summary = {}
        
        for model_name, results in all_results.items():
            summary[model_name] = {
                "avg_semantic_similarity": results["semantic_similarity"].mean(),
                "avg_lexical_diversity": results["lexical_diversity"].mean(),
                "avg_inference_time": results["inference_time"].mean()
            }
        
        summary_df = pd.DataFrame(summary).T
        
        # Create visualizations
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Semantic similarity
        sns.barplot(x=summary_df.index, y="avg_semantic_similarity", data=summary_df, ax=axes[0])
        axes[0].set_title("Average Semantic Similarity")
        axes[0].set_ylim(0, 1)
        
        # Lexical diversity
        sns.barplot(x=summary_df.index, y="avg_lexical_diversity", data=summary_df, ax=axes[1])
        axes[1].set_title("Average Lexical Diversity")
        axes[1].set_ylim(0, 1)
        
        # Inference time
        sns.barplot(x=summary_df.index, y="avg_inference_time", data=summary_df, ax=axes[2])
        axes[2].set_title("Average Inference Time (seconds)")
        
        plt.tight_layout()
        plt.savefig("model_comparison.png")
        plt.show()
        
        return summary_df
    
    def save_sample_outputs(self, all_results, num_examples=5):
        """Save sample outputs from each model for qualitative analysis."""
        sample_outputs = {}
        
        for model_name, results in all_results.items():
            # Sort by semantic similarity and take examples with good similarity
            good_samples = results.sort_values("semantic_similarity", ascending=False).head(num_examples)
            sample_outputs[f"{model_name}_good"] = good_samples[["original", "paraphrase", "semantic_similarity", "lexical_diversity"]]
            
            # Also take some examples with high lexical diversity
            diverse_samples = results.sort_values("lexical_diversity", ascending=False).head(num_examples)
            sample_outputs[f"{model_name}_diverse"] = diverse_samples[["original", "paraphrase", "semantic_similarity", "lexical_diversity"]]
        
        # Save to CSV
        for name, samples in sample_outputs.items():
            samples.to_csv(f"{name}_samples.csv", index=False)
            
        return sample_outputs

# Example usage
if __name__ == "__main__":
    # Initialize evaluator
    evaluator = ParaphraseEvaluator()
    
    # Define models to compare
    model_configs = {
        "T5-small": "t5-small",
        "BART-base": "facebook/bart-base",
        "DistilBART": "sshleifer/distilbart-cnn-6-6"
    }
    
    # Compare models (reduce num_samples if needed for memory/time constraints)
    results = evaluator.compare_models(model_configs, num_samples=50)
    
    # Visualize results
    summary = evaluator.visualize_results(results)
    print(summary)
    
    # Save sample outputs
    samples = evaluator.save_sample_outputs(results)