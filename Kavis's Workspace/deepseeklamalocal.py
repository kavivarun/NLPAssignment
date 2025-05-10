from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# ── Setup ─────────────────────────────────────────────────────────
model_id    = "microsoft/phi-3-mini-4k-instruct"
device      = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if device == "cuda" else torch.float32

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = (AutoModelForCausalLM
         .from_pretrained(model_id,
                          torch_dtype=torch_dtype,
                          device_map="auto")
         .eval())

messy_text = """
okay so like yesterday i was walking down the street and I seen this dog who was like barking so loud that i think my ears was gonna explode or maybe fall off who knows but anyway i keep walking and then this guy come up to me and he say hey do you know where the mall is at and I’m like bro why you asking me i don’t even look like i’m from here because like i just moved here like three days ago...
"""

stop_token = "###END"              # something the rewrite will never contain

prompt = (
    "Rewrite the following text to make it clearer, polished, and formal. "
    "Output **only** the rewritten passage, then write '" + stop_token + "'.\n\n"
    + messy_text + "\n"
)

# ── Tokenise ──────────────────────────────────────────────────────
inputs      = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
prompt_len  = inputs.input_ids.shape[-1]

# ── Generate ──────────────────────────────────────────────────────
with torch.no_grad():
    out_ids = model.generate(
        **inputs,
        max_new_tokens=300,
        temperature=0.1,        # very conservative
        top_p=0.9,
        do_sample=True,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )[0]

# ── Extract rewritten passage only ───────────────────────────────
text = tokenizer.decode(out_ids[prompt_len:], skip_special_tokens=True)
clean = text.split(stop_token)[0].strip()

print(clean)
