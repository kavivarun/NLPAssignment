from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    torch_dtype="auto",
    device_map="auto"
)

# Updated system message
messages = [
    {
        "role": "system",
        "content": "make the language for this better."
    },
    {
        "role": "user",
        "content": (
            "so like yesterday i was tryna go to the mall but like i aint got no ride and my friend "
            "was supposed to pick me up but like he was late as always and then when we finally got there "
            "and we didnt have no umbrella so we got all wet and cold but whatever cuz it was kinda funny honestly "
            "and then when i got home my mom was like why you so wet and i was like dont worry bout it it was just rain lol"
        ),
    },
]

# Apply chat template
prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# Generate output
outputs = pipe(prompt, max_new_tokens=512, do_sample=True, temperature=0.1, top_k=50, top_p=0.95)

# Print the clean rewritten text
print(outputs[0]["generated_text"])
