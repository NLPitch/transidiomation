import logging

import regex as re
from openai import OpenAI
from langdetect import detect
from easygoogletranslate import EasyGoogleTranslate
from utils.load_data import *

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

translator = EasyGoogleTranslate()

def baseline_google(input_sentence:str, target_language_iso:str) -> str:
    return translator.translate(f'{input_sentence}', target_language=target_language_iso)

def naiveGPT(input_sentence:str, target_language:str, api_key:str) -> str:
    client = OpenAI(
        api_key=api_key
    )

    completion = client.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=[
            {"role": "user", "content": f"Translate {input_sentence} to {target_language}"}
        ]
    )

    return str(completion.choices[0].message.content)

def transidiomation(input_sentence:str, target_language:str, api_key:str) -> str:
    if detect(input_sentence) == 'ko':
        return transidiomation_ko_2shot(input_sentence, target_language, api_key)
    
    client = OpenAI(
        api_key=api_key
    )

    completion = client.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=[
            {"role": "system", "content": f'''
                Translate the sentence to {target_language} following these steps:
                Step 1. Identify the idiom
                Step 2. Find an idiom with the same meaning in the target language. If there is no equivalent idiom, give the figurative meaning of the expression.
                Step 3. Include answer from Step 2 to translate the sentence.
            '''},
            {"role": "user", "content": f"{input_sentence}"}
        ]
    )

    return re.split(r'Step\s*3', str(completion.choices[0].message.content))[1]

def transidiomation_ko_2shot(input_sentence:str, target_language:str, api_key:str) -> str:
    client = OpenAI(
        api_key=api_key
    )

    message = []
    message.append({"role": "system", "content": f'''
                        Translate the sentence to {target_language} following these steps:
                        Step 1. Identify the idiom
                        Step 2. Find an idiom with the same meaning in the target language. If there is no equivalent idiom, give the figurative meaning of the expression.
                        Step 3. Include answer from Step 2 to translate the sentence.
                    '''})
    examples = load_data('./rsrc/korean_2shot_example.pkl')
    message.extend(examples)
    message.append({"role": "user", "content": f"{input_sentence}"})

    completion = client.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=message
    )

    return re.split(r'Step\s*3', str(completion.choices[0].message.content))[1]