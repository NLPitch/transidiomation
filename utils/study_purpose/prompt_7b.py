import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.load_data import *

# Mistral Model
model = AutoModelForCausalLM.from_pretrained(
    'mistralai/Mistral-7B-Instruct-v0.2',
    device_map = 'auto'
).to(device)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

def prompt_mistral_naive(input_sentence:str, target_language:str) -> str:
    messages = [{"role": "user", "content": f"Translate {input_sentence} to {target_language}"}]

    encoded = tokenizer.apply_chat_template(messages, return_tensors="pt")
    model_inputs = encoded.to(device)

    generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
    decoded = tokenizer.batch_decode(generated_ids)

    return decoded[0].split('[/INST]')[1]

def prompt_mistral_transidiomation(input_sentence:str, target_language:str) -> str:
    message = [
        {"role": "user", "content": f'''
            Translate the sentence '원숭이도 나무에서 떨어질 때가 있나 보다.' to {target_language} following these steps:
            Step 1. Identify the idiom
            Step 2. Find an idiom with the same meaning in the target language. If there is no equivalent idiom, give the figurative meaning of the expression.
            Step 3. Include answer from Step 2 to translate the sentence.
        '''},
        {'role': 'assistant', 'content': 'Step 1. 원숭이도 나무에서 떨어진다 Step 2. Even Homer sometimes nod Step 3. I guess even Homer sometimes nods.'},

        {"role": "user", "content": f'''
            Translate the sentence '그 여자는 내 남자친구에게 꼬리를 쳤다.' to {target_language} following these steps:
            Step 1. Identify the idiom
            Step 2. Find an idiom with the same meaning in the target language. If there is no equivalent idiom, give the figurative meaning of the expression.
            Step 3. Include answer from Step 2 to translate the sentence.
        '''},
        {'role': 'assistant', 'content': 'Step 1. 꼬리를 치다 Step 2. To flirt Step 3. She’s always flirting with my boyfriend!'},

        {"role": "user", "content": f'''
            Translate the sentence '{input_sentence}' to {target_language} following these steps:
            Step 1. Identify the idiom
            Step 2. Find an idiom with the same meaning in the target language. If there is no equivalent idiom, give the figurative meaning of the expression.
            Step 3. Include answer from Step 2 to translate the sentence.
        '''},
    ]

    encoded = tokenizer.apply_chat_template(message, return_tensors="pt")
    model_inputs = encoded.to(device)

    generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
    decoded = tokenizer.batch_decode(generated_ids)

    return decoded[0]