import os

import torch
import evaluate

import numpy as np
import matplotlib.pyplot as plt

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(
    api_key = os.getenv("OPENAI_API_KEY")
)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

google_bleu = evaluate.load("google_bleu")
bleurt = evaluate.load("bleurt", module_type="metric")
meteor = evaluate.load('meteor')
rouge = evaluate.load('rouge')

def evaluate_with_google_bleu(struct_input:dict, reference_col:str, prediction_col:str):
    list_prediction = [str(struct_input[prediction_col])]
    list_reference = [[str(struct_input[reference_col])]]

    result = google_bleu.compute(predictions=list_prediction, references=list_reference)

    return result['google_bleu']

def evaluate_with_bleurt(struct_input:dict, reference_col:str, prediction_col:str):
    list_prediction = [str(struct_input[prediction_col])]
    list_reference = [str(struct_input[reference_col])]

    results = bleurt.compute(predictions=list_prediction, references=list_reference)

    return results['scores']

def evaluate_with_meteor(struct_input:dict, reference_col:str, prediction_col:str):
    list_prediction = [str(struct_input[prediction_col])]
    list_reference = [str(struct_input[reference_col])]

    results = meteor.compute(predictions=list_prediction, references=list_reference)

    return results['meteor']

def evaluate_with_rouge(struct_input:dict, reference_col:str, prediction_col:str):
    list_prediction = [str(struct_input[prediction_col])]
    list_reference = [str(struct_input[reference_col])]

    results = rouge.compute(predictions=list_prediction, references=list_reference)

    return results['rougeL']

def evaluation_by_openai(struct_input:dict, reference_col:str, prediction_col:str) -> str:
    prediction = [str(struct_input[prediction_col])]
    reference = [str(struct_input[reference_col])]

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role":"system", "content": "On a scale from 0 to 100 rate the similarity of the two sentences where 1 is being dissimilar and 10 being identical"},
            {"role":"user", "content":"1. 'I love you' 2. 'I like you'"},
            {"role":"assistant", "content":"90"},

            {"role":"user", "content":f"1. 'I love you' 2. 'I want to eat pizza tonight'"},
            {"role":"user", "content":f"0"},

            {"role":"user", "content":f"1. '{prediction}' 2. '{reference}'"}
        ]
    )

    return completion.choices[0].message.content

def box_and_whisker(list_scores:list, plot_name:str):
    arr_scores = np.array(list_scores)

    q1 = np.quantile(arr_scores, 0.25)
    q3 = np.quantile(arr_scores, 0.75)
    med = np.median(arr_scores)

    iqr = q3 - q1
    upper_bound = q3 + (1.5 * iqr)
    lower_bound = q1 - (1.5 * iqr)

    outliers = arr_scores[(arr_scores <= lower_bound) | (arr_scores >= upper_bound)]

    plt.boxplot(arr_scores, vert=0)
    # fig = plt.figure(figsize =(10, 7))
    plt.savefig(f'{plot_name}.png')