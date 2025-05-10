from transformers import AutoTokenizer, T5ForConditionalGeneration
import nltk

# Download NLTK sentence tokenizer
from nltk.tokenize import sent_tokenize

# Load model and tokenizer
model_id = "grammarly/coedit-large"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = T5ForConditionalGeneration.from_pretrained(model_id)

# Your long input text
long_text = """
okay so like yesterday i was walking down the street and I seen this dog who was like barking so loud that i think my ears was gonna explode or maybe fall off who knows but anyway i keep walking and then this guy come up to me and he say hey do you know where the mall is at and I’m like bro why you asking me i don’t even look like i’m from here because like i just moved here like three days ago...
"""

# Split text into sentences
sentences = sent_tokenize(long_text)

# Combine sentences into chunks that fit the model's max token limit
chunks = []
current_chunk = ""
for sentence in sentences:
    tokens = tokenizer(current_chunk + " " + sentence, return_tensors="pt").input_ids.shape[1]
    if tokens < 512:
        current_chunk += " " + sentence
    else:
        chunks.append(current_chunk.strip())
        current_chunk = sentence
if current_chunk:
    chunks.append(current_chunk.strip())

# Process each chunk: paraphrase + improve
edited_chunks = []
for chunk in chunks:
    input_text = f"Paraphrase and improve the clarity, style, and grammar of the following text: {chunk}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(
        inputs.input_ids,
        max_length=512,
        do_sample=True,
        top_p=0.9,
        temperature=0.7
    )
    edited_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    edited_chunks.append(edited_text)

# Combine improved and paraphrased chunks
final_text = " ".join(edited_chunks)
print(final_text)
