import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import polars as pl

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = AutoModelForCausalLM.from_pretrained(
    'mistralai/Mistral-7B-Instruct-v0.1',
    device_map = 'auto'
).to(device)

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")

# You can have multiple user, assistant pairs to act as few-shot prompting
messages = [
    {"role": "user", "content": "This should be the question portion?"},
    {"role": "assistant", "content": "This would be the answer portion"},
]

encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

model_inputs = encodeds.to(device)

generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
decoded = tokenizer.batch_decode(generated_ids)

print(decoded[0])