import os
from glob import glob
from pathlib import Path
import torch
import json
import pandas as pd

# This script is built on top of talkative-llm wrapper and refers to usage.py script in the talkative-llm repo. More information in the talkative-llm repository repository: https://github.com/minnesotanlp/talkative-llm
from  talkative_llm.llm import (AlpacaLoraCaller, CohereCaller,
                               HuggingFaceCaller, MPTCaller, OpenAICaller)

CONFIG_DIR = Path(__file__).parent.parent
def openai_caller_completion(prompts,file_path):
    config_path = CONFIG_DIR / 'talkative-llm' / 'configs' / 'openai' / 'openai_completion_example.yaml'
    caller = OpenAICaller(config=config_path)
    results = caller.generate(prompts)
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(results, file, ensure_ascii=False, indent=4)
    print(f"Data written to {file_path} successfully.")
    del caller


if __name__=="__main__":
    
    
    # Uncomment for NAIVE PROMPTING
    # head_prompt=["Translate the sentence into Spanish."]
    # file_path = ["./data/json/spanish_naive_prompt_output.json"]


    # Uncomment for TRANSIDIOMATION PROMPTING
    head_prompt = ["Translate the sentences into Spanish by following these steps:\nStep 1. Identify the idiom.\nStep 2. Find an idiom with the same meaning in the target language. If there is no equivalent idiom, give the figurative meaning of it. If there is no equivalent idiom, translate the idiom according to its meaning.\nStep 3. Include response from Step 2 to translate the sentence., User: 'When I learn Spanish at the Spanish Academy exams are a piece of cake.'\nAssistant: Step 1. The idiom in the sentence is 'a piece of cake'\n Step 2. The idiom translates to 'pan comido'\nStep3. Cuando aprendo espanol en 'The Spanish Academy' los examenes son pan comido. , User: 'Whenever I try to concentrate in math class, I end up having my head in the clouds.'\nAssistant: Step 1. The idiom in the sentence is 'head in the clouds'\nStep 2. The idiom translates to 'estar en las nubes'\nStep 3. Siempre que intento concentrarme en la clase de matemáticas, termino estando en las nubes."]
    file_path = ["./data/json/spanish_transidiomation_prompt_output.json"]

    # Uncomment for TWO SHOT PROMPTING
    # head_prompt = ["User: 'When I learn Spanish at the Spanish Academy exams are a piece of cake.'\nAssistant: Step 1. The idiom in the sentence is 'a piece of cake'\n Step 2. The idiom translates to 'pan comido'\nStep3. Full sentence translates to 'Cuando aprendo espanol en 'The Spanish Academy' los examenes son pan comido.' , User: 'Whenever I try to concentrate in math class, I end up having my head in the clouds.'\nAssistant: Step 1. The idiom in the sentence is 'head in the clouds'\nStep 2. The idiom translates to 'estar en las nubes'\nStep 3. Full sentence translates to 'Siempre que intento concentrarme en la clase de matemáticas, termino estando en las nubes.'"]
    # file_path = ["./data/json/spanish_two_shot_output.json"]
    
    data = pd.read_csv('./data/csv/spanish_idioms_dataset.csv')
    if " Idiom embedded in a sentence (English)" in data.columns:
        english_idioms = data[" Idiom embedded in a sentence (English)"].tolist()
    else:
        print("ERROR: Check Column Name")
    sentences = english_idioms
    prompts=[]
    for method in range(len(head_prompt)):
        for sentence in sentences:
            prompts.append(head_prompt[method]+' '+sentence)
        openai_caller_completion(prompts, file_path[method])
        
        



