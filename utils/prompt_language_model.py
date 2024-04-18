import os
from dotenv import load_dotenv
from openai import OpenAI

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.load_data import load_data
from utils.generate_input_prompt import create_prompt_to_model

load_dotenv()

client = OpenAI(
    api_key = os.getenv("OPENAI_API_KEY")
)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

# input_message = load_data('/home/yaoyi/pyo00005/CSci5541/transidiomation/rsrc/inputs.pkl')

def prompt_openai(source_text:str) -> str:
  # message = input_message[mssg_type]

  completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    # messages=[
    #   {"role": "system", "content": '''Translate the sentence to English following these steps: 
    #    Step 1. Identify the idiom 
    #    Step 2. Find an idiom with the same meaning in the target language. If there is no equivalent idiom, give the figurative meaning of the expression.
    #    Step 3. Include answer from step 2 to translate the sentence.
    #    '''},
    #   {"role": "user", "content": f"{source_text}"}
    # ]
    # messages=[
    #   {"role": "user", "content": "Translate 원숭이도 나무에서 떨어질 때가 있나 보다."},
    #   {"role": "assistant", "content": "I guess even Homer sometimes nods."},
    #   {"role": "user", "content": "Translate 그 여자는 내 남자친구에게 꼬리를 쳤다"},
    #   {"role": "assistant", "content": "She’s always flirting with my boyfriend!"},
    #   {"role": "user", "content": "Translate 친구는 계속 새 차를 사라고 나에게 바람을 넣었다"},
    #   {"role": "assistant", "content": "My friend is always going on at me to buy a new car"},
    #   {"role": "user", "content": "Translate 고양이한테 생선 가게를 맡긴 꼴이 되었다"},
    #   {"role": "assistant", "content": "It's like having the fox guard the henhouse."},
    #   {"role": "user", "content": "Translate ‘개천에서 용 났다’라는 말은 강의 인생 여정을 잘 설명해 줄 수 있다."},
    #   {"role": "assistant", "content": "The term ‘rags to riches’ best describes the process for Kang."},
    #   {"role": "user", "content": f'Translate {source_text} to English'},
    # ]
    messages=[
      {"role": "system", "content": '''Translate the sentence to English following these steps: 
       Step 1. Identify the idiom 
       Step 2. Find an idiom with the same meaning in the target language. If there is no equivalent idiom, give the figurative meaning of the expression.
       Step 3. Include answer from step 2 to translate the sentence.
       '''},
      {"role": "user", "content": "원숭이도 나무에서 떨어질 때가 있나 보다."},
      {"role": "assistant", "content": "Step 1. 원숭이도 나무에서 떨어진다 Step 2. Even Homer sometimes nod Step 3. I guess even Homer sometimes nods."},
      {"role": "user", "content": "그 여자는 내 남자친구에게 꼬리를 쳤다"},
      {"role": "assistant", "content": "Step 1. 꼬리를 치다 Step 2. To flirt Step 3. She’s always flirting with my boyfriend!"},
      {"role": "user", "content": "친구는 계속 새 차를 사라고 나에게 바람을 넣었다"},
      {"role": "assistant", "content": "Step 1. 바람을 넣다 Step 2. Motivate Step 3. My friend is always going on at me to buy a new car"},
      {"role": "user", "content": "고양이한테 생선 가게를 맡긴 꼴이 되었다"},
      {"role": "assistant", "content": "Step 1. 고양이에게 생선을 맡기다 Step 2. Fox guard the henhouse Step 3. It's like having the fox guard the henhouse."},
      {"role": "user", "content": "‘개천에서 용 났다’라는 말은 강의 인생 여정을 잘 설명해 줄 수 있다"},
      {"role": "assistant", "content": "Step 1. 개천에서 용 났다 Step 2. Rags to riches Step 3. The term ‘rags to riches’ best describes the process for Kang."},
      {"role": "user", "content": f"{source_text}"}
    ]
  )

  print(completion.choices[0].message.content)

  return str(completion.choices[0].message.content)