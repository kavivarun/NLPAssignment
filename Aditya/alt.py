import torch
import time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load similarity model for evaluation
sim_model = SentenceTransformer('all-MiniLM-L6-v2').to(device)

# Function to calculate semantic similarity
def calculate_similarity(text1, text2):
    embedding1 = sim_model.encode(text1, convert_to_tensor=True)
    embedding2 = sim_model.encode(text2, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(embedding1, embedding2).item()
    return similarity

# Function to calculate lexical diversity
def calculate_diversity(original, paraphrase):
    # Simple tokenization by splitting
    original_tokens = original.lower().split()
    paraphrase_tokens = paraphrase.lower().split()
    
    # Calculate word overlap (Jaccard similarity)
    original_set = set(original_tokens)
    paraphrase_set = set(paraphrase_tokens)
    
    if len(original_set.union(paraphrase_set)) > 0:
        jaccard = len(original_set.intersection(paraphrase_set)) / len(original_set.union(paraphrase_set))
    else:
        jaccard = 0
        
    # Higher score means more diverse
    diversity_score = 1 - jaccard
    return diversity_score

# Load models
models = {}
tokenizers = {}

model_configs = {
    "T5-small": {
        "model_id": "t5-small",
        "prefix": "paraphrase: "
    },
    "BART-base": {
        "model_id": "facebook/bart-base",
        "prefix": ""
    },
    "DistilBART": {
        "model_id": "sshleifer/distilbart-cnn-6-6",
        "prefix": ""
    }
}

for model_name, config in model_configs.items():
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(config["model_id"])
    model = AutoModelForSeq2SeqLM.from_pretrained(config["model_id"]).to(device)
    
    models[model_name] = model
    tokenizers[model_name] = tokenizer
    print(f"Successfully loaded {model_name}")

# Function to generate paraphrase
def generate_paraphrase(model_name, text):
    model = models[model_name]
    tokenizer = tokenizers[model_name]
    prefix = model_configs[model_name]["prefix"]
    
    input_text = prefix + text
    inputs = tokenizer(input_text, return_tensors="pt", max_length=128, truncation=True).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=128,
            num_beams=4,
            early_stopping=True
        )
    
    paraphrase = tokenizer.decode(outputs[0], skip_special_tokens=True)
    similarity = calculate_similarity(text, paraphrase)
    diversity = calculate_diversity(text, paraphrase)
    
    return paraphrase, similarity, diversity

# Test sentences covering different styles and complexity
test_sentences = [
    "The company announced record profits in the last quarter.",
    "Scientists discovered a new species of deep-sea fish that can survive extreme pressure.",
    "Could you please explain how this algorithm works in simpler terms?",
    "The patient showed significant improvement after receiving the experimental treatment.",
    "The integration of artificial intelligence in healthcare has raised both hopes and concerns among medical professionals."
]

# Generate paraphrases for each test sentence using each model
results = []

print("\n=== Sample Paraphrases ===\n")
for i, sentence in enumerate(test_sentences):
    print(f"Original ({i+1}): {sentence}")
    print("-" * 80)
    
    for model_name in models.keys():
        paraphrase, similarity, diversity = generate_paraphrase(model_name, sentence)
        
        print(f"{model_name}:")
        print(f"  Paraphrase: {paraphrase}")
        print(f"  Semantic Similarity: {similarity:.4f}")
        print(f"  Lexical Diversity: {diversity:.4f}")
        print()
        
    print("=" * 80)
    print()

