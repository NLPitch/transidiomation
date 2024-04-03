import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from generate_input_prompt import create_prompt_to_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Mistral Models
model = AutoModelForCausalLM.from_pretrained(
    'mistralai/Mistral-7B-Instruct-v0.2',
    device_map = 'auto'
).to(device)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

def prompt_mistral(source_text):
    messages = create_prompt_to_model(0, source_text, 'Korean', 'English')

    encoded = tokenizer.apply_chat_template(messages, return_tensors="pt")
    model_inputs = encoded.to(device)

    generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
    decoded = tokenizer.batch_decode(generated_ids)

    return decoded[0]

def prompt_gemma():
    model = "google/gemma-7b-it"
    return 0