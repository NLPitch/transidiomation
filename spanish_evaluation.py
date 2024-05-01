
from jiwer import wer
import json
from rouge import Rouge
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import pandas as pd
import csv
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction,sentence_bleu
import openai
import os
from scipy.spatial.distance import cosine
import tensorflow as tf
from bleurt import score

openai.api_key = 'openai-key'
file_path = './data/csv/spanish_idioms_dataset.csv'  # Replace 'your_file.csv' with the path to your CSV file
data = pd.read_csv(file_path)

def calculate_similarity(embedding1, embedding2):
    return 1 - cosine(embedding1, embedding2)

def get_embedding(text):
    response = openai.Embedding.create(
        input=[text],
        engine="text-embedding-3-large")
    return response['data'][0]['embedding']


if "idiom embedded in a sentence (Spanish)" in data.columns:
    spanish_idioms = data["idiom embedded in a sentence (Spanish)"].tolist()
else:
    print("Column not found. Please check the column name and try again.")  
reference = spanish_idioms
methods = ["Google Machine Translation", "Naive Prompt Metrics: ","Proposed Method Metrics: ","Two-shot Metrics: "]
files = ["./data/csv/google_spanish_machine_translation.csv", "./data/json/spanish_naive_prompt_output.json","./data/json/spanish_transidiomation_prompt_output.json", "./data/json/spanish_two_shot_prompt_output.json"]

all_rouge_scores=[]
all_wer_scores=[]
all_bleu_scores = []
all_openai_scores=[]
all_google_bleurt_scores=[]



for prompt_no, file in enumerate(files):
    print(methods[prompt_no])
    hypothesis = []
    # Google translation
    if not prompt_no:
        with open(file,"r") as file:
            data = pd.read_csv(file)
        for row in data["Machine Translated by Google"]:    
            hypothesis.append(row)
    
    else:
        with open(file,'r') as file:  
            data = json.load(file)
        # Naive
        if prompt_no == 1:
            for obj in data:
                modified = obj["generation"].replace("\n", "")
                hypothesis.append(modified)
        
        # Proposed Method
        elif prompt_no == 2:
            for obj in data:
                modified = obj["generation"].replace("\n", "")
                if "3. " in modified:
                    modified = modified.split("3. ")[1]
                    modified = modified.replace("Spanish translation: ", "")
                    modified = modified.replace("Give a Spanish translation of the sentence including that idiom: ", "")
                    modified = modified.replace("Spanish translation of the sentence:","")
                    modified = modified.replace('"', "")
                    hypothesis.append(modified) 
                
                elif "3:" in modified:
                    modified = modified.split("3:")[1]
                    modified = modified.replace('"', "")
                    hypothesis.append(modified)          
                
                else:
                    hypothesis.append("")
        # Two shot
        else:
            for obj in data:
                modified = obj["generation"].replace("\n", "")
                
                if "Full sentence translates to " in modified:
                    modified = modified.split("Full sentence translates to")[1]
                    modified = modified.replace("'", "")
                    hypothesis.append(modified)
                
                else:
                    hypothesis.append("")
        
    # ROUGE metrics
    rouge = Rouge()
    total_scores =0.0
    rouge_f1_scores = [rouge.get_scores(hyp, ref) for hyp, ref in zip(hypothesis, reference) if len(hyp)>1]
    rouge_f1_scores = [scores[0]['rouge-1']['f'] for scores in rouge_f1_scores]
    rouge_statistics = {
        'mean': np.mean(rouge_f1_scores),
        'median': np.median(rouge_f1_scores),
        'std_deviation': np.std(rouge_f1_scores),
        
    }
    all_rouge_scores.append(rouge_f1_scores)
    
    # WER Metrics
    wers = [wer(ref, hyp) for ref, hyp in zip(reference, hypothesis) if len(hyp)>1 ]
    wer_statistics = {
        'mean': np.mean(wers),
        'median': np.median(wers),
        'std_deviation': np.std(wers),
    }
    all_wer_scores.append(wers)
    
    # BLEU Metrics
    bleu_scores_1gram = [ sentence_bleu([ref.split()], cand.split(), weights=(1, 0, 0, 0), smoothing_function=SmoothingFunction().method1)
        for ref, cand in zip(reference, hypothesis) if len(cand) > 1]
    bleu_statistics = {
        'mean': np.mean(bleu_scores_1gram),
        'median': np.median(bleu_scores_1gram),
        'std_dev': np.std(bleu_scores_1gram)
    }
    all_bleu_scores.append(bleu_scores_1gram)
    if prompt_no >= 1:
        # matching candidate and reference data due misalignment and no output
        new_reference =[]
        remove_ind = [ind for ind, hyp in enumerate(hypothesis) if len(hyp) > 1] 
        for i in range(len(hypothesis)):
            if i in remove_ind:
                new_reference.append(reference[i])
        hypothesis = [hyp for ind, hyp in enumerate(hypothesis) if len(hyp) > 1] 
        
        # Open AI embedding extraction and prep for evaluation
        candidate_embeddings = [get_embedding(text) for text in hypothesis if len(text) > 1]
        reference_embeddings = [get_embedding(text) for text in new_reference if len(text) > 1]
        print(len(new_reference), len(hypothesis))
        # GOOGLE BLEURT evaluation
        bleurt_ops = score.create_bleurt_ops()
        bleurt_out = bleurt_ops(references=tf.constant(new_reference), candidates=tf.constant(hypothesis))
        google_scores = bleurt_out["predictions"].numpy()
    else:
        # open_ai embeddings
        candidate_embeddings = [get_embedding(text) for text in hypothesis if len(text) >1]
        reference_embeddings = [get_embedding(text) for text in reference if len(text) >1]        
        # google bleurt
        bleurt_ops = score.create_bleurt_ops()
        hypothesis = [hyp for ind, hyp in enumerate(hypothesis) if len(hyp) > 1] 
        bleurt_out = bleurt_ops(references=tf.constant(reference), candidates=tf.constant(hypothesis))
        assert bleurt_out["predictions"].shape == (len(hypothesis))
        google_scores = bleurt_out["predictions"].numpy()
    # google bleurt statistics
    all_google_bleurt_scores.append(list(google_scores))
    google_statistics = {
        'mean': np.mean(google_scores),
        'median': np.median(google_scores),
        'std_dev': np.std(google_scores)
    } 

    # openai statistics
    similarities = [calculate_similarity(cand, ref) for cand, ref in zip(candidate_embeddings, reference_embeddings)]
    openai_statistics = {
        'mean': np.mean(similarities),
        'median': np.median(similarities),
        'std_dev': np.std(similarities)
    }
    all_openai_scores.append(similarities)
    print("\tRouge F1 Statistics: \n", "\t\tMean ROUGE-1 F1: ", rouge_statistics["mean"],"\n\t\tMedian ROUGE-1 F1: ",rouge_statistics["median"],"\n\t\tStd. Deviation ROUGE-1 F1: " , rouge_statistics["std_deviation"])
    print("\tWord Error Rate Statistics: \n", "\t\tMean WER: ", wer_statistics["mean"], "\n\t\tMedian WER: ", wer_statistics['median'], "\n\t\tStd. Deviation WER: ", wer_statistics['std_deviation'] )
    print("\tBLEU Statistics: \n", "\tMean BLEU: ", bleu_statistics["mean"],"\n\t\tMedian BLEU: ",bleu_statistics["median"],"\n\t\tStd. Deviation BLEU: " , bleu_statistics["std_dev"])
    print("\tOpenAI Statistics: \n", "\tMean OpenAI: ", openai_statistics["mean"],"\n\t\tMedian OpenAI: ",openai_statistics["median"],"\n\t\tStd. Deviation OpenAI: " , openai_statistics["std_dev"])
    print("\tGoogle BLEURT Statistics: \n", "\tMean Google BLEURT: ", google_statistics["mean"],"\n\t\tMedian Google BLEURT: ",google_statistics["median"],"\n\t\tStd. Deviation Google BLEURT: " , google_statistics["std_dev"])
    print("\tError Percentage: ",((len(data)-len(rouge_f1_scores))/len(data))*100, "%")
    
    
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'  # Specify the sans-serif font you want to use
p_value_lst = []
metric_score_lst= {"BLEURT":all_google_bleurt_scores,"BLEU":all_bleu_scores,"ROUGE":all_rouge_scores,"WER":all_wer_scores, "OpenAI":all_openai_scores}
keys = list(metric_score_lst.keys())
# Generating stats and graphs for all metrics
# Check out the graph in data/stats/spanish_{evaluation metric}.png
for i in range(len(list(metric_score_lst.values()))):
    print(keys[i])
    scores = [score for sublist in metric_score_lst[keys[i]][:3] for score in sublist]
    methods_json = ["Google Translation", "Naive Prompting", "Proposed Method"]
    labels = [methods_json[i] for i, sublist in enumerate(metric_score_lst[keys[i]][:3]) for _ in sublist]
    # Create a single dataframe for both datasets
    scores_total = scores
    labels_total = labels
    dataset = ['Spanish']*len(scores_total)
    df = pd.DataFrame({'Scores': scores_total, 'Method': labels_total, 'Dataset': dataset})
    # Create the plot
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='Scores', y='Method', hue='Dataset', data=df, orient='h')
    plt.title('Distribution of Evaluation Scores by Prompting Methods', fontsize=20)
    plt.ylabel('Method', fontsize=16)
    plt.legend(title='Dataset')
    plt.savefig(f'./output/stats/spanish_{keys[i]}.png')
    google_scores = df['Scores'][df['Method'] == 'Google Translation']
    naive_scores = df['Scores'][df['Method'] == 'Naive Prompting']
    step_scores = df['Scores'][df['Method'] == 'Proposed Method']
    all_scores = [list(google_scores), list(naive_scores), list(step_scores)]
    count =0
    total_scores =0
    stat, p_value_step_google = stats.mannwhitneyu(step_scores, google_scores, alternative='two-sided')
    stat, p_value_naive_google = stats.mannwhitneyu(naive_scores, google_scores, alternative='two-sided')
    stat, p_value_step_naive = stats.mannwhitneyu(step_scores, naive_scores, alternative='two-sided')
    print(f"P-value for Proposed Method vs. Google Translation: {p_value_step_google:.3f}")
    print(f"P-value for Naive vs. Google Translation: {p_value_naive_google:.3f}")
    print(f"P-value for Proposed Method vs. Naive: {p_value_step_naive:.3f}")
    p_value_lst.append([p_value_step_naive, p_value_naive_google, p_value_step_google])


data = {
    "Comparison": [
        "Naive vs. Proposed Method", "Google Translation vs. Naive", "Google Translation vs. Proposed Method",

        "Naive vs. Proposed Method", "Google Translation vs. Naive", "Google Translation vs. Proposed Method",

        "Naive vs. Proposed Method", "Google Translation vs. Naive", "Google Translation vs. Proposed Method",

        "Naive vs. Proposed Method", "Google Translation vs. Naive", "Google Translation vs. Proposed Method",

        "Naive vs. Proposed Method", "Google Translation vs. Naive", "Google Translation vs. Proposed Method"
    ],
    "Metric": [
        "OpenAI", "OpenAI", "OpenAI",
        "BLEURT", "BLEURT",  "BLEURT",
        "ROUGE","ROUGE","ROUGE",
        "WER","WER","WER",
        "BLEU","BLEU","BLEU",
        
    ],
    "P-Value": [
        p_value_lst[4][0],p_value_lst[4][1],p_value_lst[4][2], 
        p_value_lst[0][0],p_value_lst[0][1],p_value_lst[0][2],
        p_value_lst[2][0], p_value_lst[2][1],p_value_lst[2][2],
        p_value_lst[3][0], p_value_lst[3][1], p_value_lst[3][2],
        p_value_lst[1][0],p_value_lst[1][1],p_value_lst[1][2],
    ]
}


# Statistical signigificance test between prompting methods based on evaluation metrics
# Check out the graph in output/stats/spanish_p_values.png
df = pd.DataFrame(data)
sns.set(style="whitegrid")
plt.figure(figsize=(12, 6))
plot = sns.barplot(x='Comparison', y='P-Value', hue='Metric', data=df, palette='coolwarm')
plt.axhline(0.05, color='red', linewidth=2, linestyle='--')
plot.text(2.5, 0.15, 'Significance threshold (p=0.05)', color = 'red', va='center', ha='center',fontsize=17)
plt.title('Statiscal Significance in Evaluation Metrics between Prompting Methods', fontsize=16)
plt.ylabel('P-Value', fontsize=14)
plt.xlabel('Comparison Methods', fontsize=14)
plt.xticks(rotation=45)  # Rotate x-labels for better readability
plt.tight_layout()
plt.savefig(f'./output/stats/spanish_p_values.png')
# Final graphs for korean and spanish
metric_files =["./data/csv/kr_bleurt.csv", "./data/csv/kr_rouge.csv"]
korean_metric_score_lst = []
# each evaluation method has four lists that corresponds to the prompting method 

for fp in metric_files:
    with open(fp) as file:
        korean_metric_score_lst.append(pd.read_csv(file))
        
        
metric_score_lst= {"BLEURT":all_google_bleurt_scores,"ROUGE":all_rouge_scores}
# need to add "WER":all_wer_scores
keys = list(metric_score_lst.keys())
for i in range(len(list(metric_score_lst.values()))):
    
    scores = [score for sublist in metric_score_lst[keys[i]][:3] for score in sublist]
    methods = ["Google Translation", "Naive Prompting", "Proposed Method"]
    labels = [methods[i] for i, sublist in enumerate(metric_score_lst[keys[i]][:3]) for _ in sublist]
    # Create a single dataframe for both datasets

    # Prepare the Korean scores
    google_kr = korean_metric_score_lst[i]["google_translate"].tolist()
    naive_kr = korean_metric_score_lst[i]['naive_gpt'].tolist()
    trans_kr = korean_metric_score_lst[i]["transidiomation"].tolist()
    # Group the Korean scores
    all_google_bleurt_scores_kr = [google_kr, naive_kr, trans_kr]
    labels_kr = [methods[i] for i, sublist in enumerate(all_google_bleurt_scores_kr) for _ in sublist]
    scores_kr = [score for sublist in all_google_bleurt_scores_kr for score in sublist]
    # Create a single dataframe for both datasets
    scores_total = scores_kr + scores
    labels_total = labels_kr + labels
    
    dataset = ['Korean']*len(scores_kr) + ['Spanish']*len(scores)
    df = pd.DataFrame({'Scores': scores_total, 'Method': labels_total, 'Dataset': dataset})
    all_scores = [google_kr, naive_kr, trans_kr]
# Calculate Q1 and Q3
    count =0
    total_scores =0
    for scores in all_scores:
        total_scores += len(scores)
        scores = pd.DataFrame(scores)
        Q1 = scores.quantile(0.25)
        Q3 = scores.quantile(0.75)
        # Calculate the IQR
        IQR = Q3 - Q1
        # Define bounds for outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        # Identify outliers
        method_lower_outliers = (scores < lower_bound)
        method_upper_outliers = (scores > upper_bound)
        
        # Append to the outliers DataFrame
        count += method_lower_outliers.iloc[:, 0].sum()
        count += method_upper_outliers.iloc[:, 0].sum()
    print(keys[i] ," ERROR RATE: ",count/total_scores)
    # Create the plot
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='Scores', y='Method', hue='Dataset', data=df, orient='h')
    plt.title(f'Distribution of Evaluation Score by Prompting Methods')
    plt.xlabel(f'{keys[i]} Score', fontsize=20)
    plt.ylabel('Method', fontsize=20)
    plt.legend(title='Dataset')
    plt.savefig(f'./output/stats/results_spain_korea_{keys[i]}.png')