from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import os

# Print current working directory
print("Current working directory:", os.getcwd())

# Load the fine-tuned model and tokenizer
model_path = "./checkpoint-1000"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Setup the text generation pipeline
text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

# Generate text based on a prompt
prompt = "Describe the benefits of using AI in modern healthcare."
generated_text = text_generator(prompt, max_length=50, num_return_sequences=1)

print(generated_text[0]['generated_text'])
