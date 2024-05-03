from transformers import AutoModelForCausalLM, AutoTokenizer
# # Mistral Model
# model = AutoModelForCausalLM.from_pretrained(
#     'mistralai/Mistral-7B-Instruct-v0.2',
#     device_map = 'auto'
# ).to(device)
# tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

# def prompt_mistral(source_text:str) -> str:
#     messages = [{"role": "user", "content": f"Identify the idiom in {source_text}"}]
#     # messages = [{"role": "user", "content": f"Translate {source_text} from Korean to English"}]

#     # messages = create_prompt_to_model(0, source_text, 'Korean', 'English')

#     encoded = tokenizer.apply_chat_template(messages, return_tensors="pt")
#     model_inputs = encoded.to(device)

#     generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
#     decoded = tokenizer.batch_decode(generated_ids)

#     return decoded[0].split('[/INST]')[1]
# # .split('[/INST]')
# # tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b-it")
# # model = AutoModelForCausalLM.from_pretrained("google/gemma-7b-it", device_map="auto").to(device)

# def prompt_gemma(source_text:str) -> str:
#     # messages = [{"role": "user", "content": f"Translate {source_text} from Korean to English"}]
#     messages = [{"role": "user", "content": f"Identify the idiom in {source_text}"}]
#     prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

#     inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
#     outputs = model.generate(input_ids=inputs.to(model.device), max_new_tokens=1000, do_sample=True)
#     decoded = tokenizer.decode(outputs[0])

#     print(decoded)

#     return decoded.split('<start_of_turn>model')[1]