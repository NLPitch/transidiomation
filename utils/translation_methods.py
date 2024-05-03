import logging

import regex as re
from openai import OpenAI
from easygoogletranslate import EasyGoogleTranslate

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