
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




def calculate_similarity(embedding1, embedding2):
    return 1 - cosine(embedding1, embedding2)
file_path = './data/csv/spanish_idioms_dataset.csv'  # Replace 'your_file.csv' with the path to your CSV file
data = pd.read_csv(file_path)

openai.api_key = 'api-key'

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
methods = ["Baseline Prompt Metrics: ","Step-by-step Metrics: ","Chain-of-thought Metrics: "]
files = ["./data/json/spanish_baseline_prompt_output.json","./data/json/spanish_step_by_step_prompt_output.json", "./data/json/spanish_cot_prompt_output.json"]
all_rouge_scores=[]
all_wer_scores=[]
all_bleu_scores = []
all_openai_scores=[]
all_google_bleurt_scores=[]
for prompt_no, file in enumerate(files):
    print(methods[prompt_no])
    with open(file,'r') as file:  
        data = json.load(file)

    hypothesis = []
    if prompt_no == 0:
        for obj in data:
            modified = obj["generation"].replace("\n", "")
            hypothesis.append(modified)

    
    elif prompt_no == 1:
        for obj in data:
            modified = obj["generation"].replace("\n", "")
            try:
                modified = modified.split("Step 3. ")[1]
                modified = modified.replace("Spanish translation: ", "")
                modified = modified.replace("Give a Spanish translation of the sentence including that idiom: ", "")
                modified = modified.replace("Spanish translation of the sentence:","")
                modified = modified.replace('"', "")
                hypothesis.append(modified)            
            except:
                hypothesis.append("")


    else:
        for obj in data:
            modified = obj["generation"].replace("\n", "")
            try:
                modified = modified.split("Step 3. ")[1]
                modified = modified.split(" '")[1]
                modified = modified.replace(".'", ".")
                hypothesis.append(modified)
                
            except:
                hypothesis.append("")
    
    
    # Initialize Rouge object
    rouge = Rouge()
    total_scores =0.0
    count_empty=0
    rouge_f1_scores = [rouge.get_scores(hyp, ref) for hyp, ref in zip(hypothesis, reference) if len(hyp)>1]
    # Detailed statistics
    rouge_f1_scores = [scores[0]['rouge-1']['f'] for scores in rouge_f1_scores]
    rouge_statistics = {
        'mean': np.mean(rouge_f1_scores),
        'median': np.median(rouge_f1_scores),
        'std_deviation': np.std(rouge_f1_scores),
        'min': np.min(rouge_f1_scores),
        'max': np.max(rouge_f1_scores),
        'percentiles': {
            '25th': np.percentile(rouge_f1_scores, 25),
            '50th': np.percentile(rouge_f1_scores, 50),
            '75th': np.percentile(rouge_f1_scores, 75)
        }
    }
    all_rouge_scores.append(rouge_f1_scores)
    count =0
    total_error = 0
    wers = [wer(ref, hyp) for ref, hyp in zip(reference, hypothesis) if len(hyp)>1 ]
    wer_statistics = {
        'mean': np.mean(wers),
        'median': np.median(wers),
        'std_deviation': np.std(wers),
        'min': np.min(wers),
        'max': np.max(wers),
        'percentiles': {
            '25th': np.percentile(wers, 25),
            '50th': np.percentile(wers, 50),
            '75th': np.percentile(wers, 75)
        }
    }
    all_wer_scores.append(wers)
    bleu_scores_1gram = [
    sentence_bleu([ref.split()], cand.split(), weights=(1, 0, 0, 0), smoothing_function=SmoothingFunction().method1)
    for ref, cand in zip(reference, hypothesis)]
    bleu_statistics = {
        'mean': np.mean(bleu_scores_1gram),
        'median': np.median(bleu_scores_1gram),
        'std_dev': np.std(bleu_scores_1gram)
    }
    all_bleu_scores.append(bleu_scores_1gram)
    
    candidate_embeddings = [get_embedding(text) for text in hypothesis if len(text) >1]
    reference_embeddings = [get_embedding(text) for text in reference if len(text) >1]
    similarities = [calculate_similarity(cand, ref) for cand, ref in zip(candidate_embeddings, reference_embeddings)]
    openai_statistics = {
        'mean': np.mean(similarities),
        'median': np.median(similarities),
        'std_dev': np.std(similarities)
    }
    all_openai_scores.append(similarities)
  
    bleurt_ops = score.create_bleurt_ops()
    reference = [ref for ref in reference if len(ref)>1]
    hypothesis = [hyp for hyp in hypothesis if len(hyp) > 1]
    bleurt_out = bleurt_ops(references=tf.constant(reference), candidates=tf.constant(hypothesis))
    assert bleurt_out["predictions"].shape == (len(hypothesis),)
    google_scores = bleurt_out["predictions"].numpy()
    all_google_bleurt_scores.append(list(google_scores))
    google_statistics = {
        'mean': np.mean(google_scores),
        'median': np.median(google_scores),
        'std_dev': np.std(google_scores)
    }    
    print("\tRouge F1 Statistics: \n", "\t\tMean ROUGE-1 F1: ", rouge_statistics["mean"],"\n\t\tMedian ROUGE-1 F1: ",rouge_statistics["median"],"\n\t\tStd. Deviation ROUGE-1 F1: " , rouge_statistics["std_deviation"])
    print("\tWord Error Rate Statistics: \n", "\t\tMean WER: ", wer_statistics["mean"], "\n\t\tMedian WER: ", wer_statistics['median'], "\n\t\tStd. Deviation WER: ", wer_statistics['std_deviation'] )
    print("\tBLEU Statistics: \n", "\tMean BLEU: ", bleu_statistics["mean"],"\n\t\tMedian BLEU: ",bleu_statistics["median"],"\n\t\tStd. Deviation BLEU: " , bleu_statistics["std_dev"])
    print("\tOpenAI Statistics: \n", "\tMean OpenAI: ", openai_statistics["mean"],"\n\t\tMedian OpenAI: ",openai_statistics["median"],"\n\t\tStd. Deviation OpenAI: " , openai_statistics["std_dev"])
    print("\tGoogle BLEURT Statistics: \n", "\tMean Google BLEURT: ", google_statistics["mean"],"\n\t\tMedian Google BLEURT: ",google_statistics["median"],"\n\t\tStd. Deviation Google BLEURT: " , google_statistics["std_dev"])
    print("\tError Percentage: ",((len(data)-len(rouge_f1_scores))/len(data))*100, "%")


mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'  # Specify the sans-serif font you want to use


scores = [score for sublist in all_google_bleurt_scores for score in sublist]
methods = ["Baseline", "Step-by-Step","Chain-of-Thought"]
labels = [methods[i] for i, sublist in enumerate(all_google_bleurt_scores) for _ in sublist]
google_df = pd.DataFrame({'Scores': scores, 'Group': labels})
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.boxplot(x='Group', y='Scores', data=google_df)
plt.title('Distribution of Google BLEURT  Scores by Prompting Method', fontsize=20)
plt.ylabel('Google BLEURT Score',fontsize =16)
plt.xlabel('Prompting Methods',fontsize =16)
plt.show()
baseline_scores = google_df['Scores'][google_df['Group'] == 'Baseline']
step_scores = google_df['Scores'][google_df['Group'] == 'Step-by-Step']
chain_scores = google_df['Scores'][google_df['Group'] == 'Chain-of-Thought']
stat, p_value_baseline_step = stats.mannwhitneyu(step_scores, baseline_scores, alternative='two-sided')
stat, p_value_baseline_chain = stats.mannwhitneyu(chain_scores, baseline_scores, alternative='two-sided')
stat, p_value_step_chain = stats.mannwhitneyu(chain_scores, step_scores, alternative='two-sided')
print(f"P-value for Baseline vs. Step-by-Step: {p_value_baseline_step:.3f}")
print(f"P-value for Baseline vs. Chain-of-Thought: {p_value_baseline_chain:.3f}")
print(f"P-value for Step-by-Step vs. Chain-of-Thought: {p_value_step_chain:.3f}")
google_p_values =[p_value_baseline_step,p_value_baseline_chain,p_value_step_chain]


scores = [score for sublist in all_bleu_scores for score in sublist]
methods = ["Baseline", "Step-by-Step","Chain-of-Thought"]
labels = [methods[i] for i, sublist in enumerate(all_bleu_scores) for _ in sublist]
wer_df = pd.DataFrame({'Scores': scores, 'Group': labels})
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.boxplot(x='Group', y='Scores', data=wer_df)
plt.title('Distribution of BLEU Scores by Prompting Method', fontsize=20)
plt.ylabel('BLEU Score',fontsize =16)
plt.xlabel('Prompting Methods',fontsize =16)
plt.show()
baseline_scores = wer_df['Scores'][wer_df['Group'] == 'Baseline']
step_scores = wer_df['Scores'][wer_df['Group'] == 'Step-by-Step']
chain_scores = wer_df['Scores'][wer_df['Group'] == 'Chain-of-Thought']
stat, p_value_baseline_step = stats.mannwhitneyu(step_scores, baseline_scores, alternative='two-sided')
stat, p_value_baseline_chain = stats.mannwhitneyu(chain_scores, baseline_scores, alternative='two-sided')
stat, p_value_step_chain = stats.mannwhitneyu(chain_scores, step_scores, alternative='two-sided')
print(f"P-value for Baseline vs. Step-by-Step: {p_value_baseline_step:.3f}")
print(f"P-value for Baseline vs. Chain-of-Thought: {p_value_baseline_chain:.3f}")
print(f"P-value for Step-by-Step vs. Chain-of-Thought: {p_value_step_chain:.3f}")
bleu_values =[p_value_baseline_step,p_value_baseline_chain,p_value_step_chain]



scores = [score for sublist in all_rouge_scores for score in sublist]
methods = ["Baseline", "Step-by-Step","Chain-of-Thought"]
labels = [methods[i] for i, sublist in enumerate(all_rouge_scores) for _ in sublist]
rouge_df = pd.DataFrame({'Scores': scores, 'Group': labels})
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.boxplot(x='Group', y='Scores', data=rouge_df)
plt.title('Distribution of ROUGE-1 F1-scores by Prompting Method', fontsize =20)
plt.ylabel('ROUGE-F1 Score',fontsize =16)
plt.xlabel('Prompting Methods',fontsize =16)
plt.show()
baseline_scores = rouge_df['Scores'][rouge_df['Group'] == 'Baseline']
step_scores = rouge_df['Scores'][rouge_df['Group'] == 'Step-by-Step']
chain_scores = rouge_df['Scores'][rouge_df['Group'] == 'Chain-of-Thought']
# Mann-Whitney U Test between each pair of methods
stat, p_value_baseline_step = stats.mannwhitneyu(step_scores, baseline_scores, alternative='two-sided')
stat, p_value_baseline_chain = stats.mannwhitneyu(chain_scores, baseline_scores, alternative='two-sided')
stat, p_value_step_chain = stats.mannwhitneyu(chain_scores, step_scores, alternative='two-sided')
print(f"P-value for Baseline vs. Step-by-Step: {p_value_baseline_step:.3f}")
print(f"P-value for Baseline vs. Chain-of-Thought: {p_value_baseline_chain:.3f}")
print(f"P-value for Step-by-Step vs. Chain-of-Thought: {p_value_step_chain:.3f}")
rouge_p_values =[p_value_baseline_step,p_value_baseline_chain,p_value_step_chain]


scores = [score for sublist in all_wer_scores for score in sublist]
methods = ["Baseline", "Step-by-Step","Chain-of-Thought"]
labels = [methods[i] for i, sublist in enumerate(all_rouge_scores) for _ in sublist]
wer_df = pd.DataFrame({'Scores': scores, 'Group': labels})
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.boxplot(x='Group', y='Scores', data=wer_df)
plt.title('Distribution of Word Error Rate Scores by Prompting Method', fontsize=20)
plt.ylabel('Word Error Rate Score',fontsize =16)
plt.xlabel('Prompting Methods',fontsize =16)
plt.show()
baseline_scores = wer_df['Scores'][wer_df['Group'] == 'Baseline']
step_scores = wer_df['Scores'][wer_df['Group'] == 'Step-by-Step']
chain_scores = wer_df['Scores'][wer_df['Group'] == 'Chain-of-Thought']
stat, p_value_baseline_step = stats.mannwhitneyu(step_scores, baseline_scores, alternative='two-sided')
stat, p_value_baseline_chain = stats.mannwhitneyu(chain_scores, baseline_scores, alternative='two-sided')
stat, p_value_step_chain = stats.mannwhitneyu(chain_scores, step_scores, alternative='two-sided')
print(f"P-value for Baseline vs. Step-by-Step: {p_value_baseline_step:.3f}")
print(f"P-value for Baseline vs. Chain-of-Thought: {p_value_baseline_chain:.3f}")
print(f"P-value for Step-by-Step vs. Chain-of-Thought: {p_value_step_chain:.3f}")
wer_p_values =[p_value_baseline_step,p_value_baseline_chain,p_value_step_chain]


scores = [score for sublist in all_openai_scores for score in sublist]
methods = ["Baseline", "Step-by-Step","Chain-of-Thought"]
labels = [methods[i] for i, sublist in enumerate(all_openai_scores) for _ in sublist]
wer_df = pd.DataFrame({'Scores': scores, 'Group': labels})
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.boxplot(x='Group', y='Scores', data=wer_df)
plt.title('Distribution of OpenAI Embedding Similarity Scores by Prompting Method', fontsize=20)
plt.ylabel('OpenAI Embedding Similarity Score',fontsize =16)
plt.xlabel('Prompting Methods',fontsize =16)
plt.show()
baseline_scores = wer_df['Scores'][wer_df['Group'] == 'Baseline']
step_scores = wer_df['Scores'][wer_df['Group'] == 'Step-by-Step']
chain_scores = wer_df['Scores'][wer_df['Group'] == 'Chain-of-Thought']
stat, p_value_baseline_step = stats.mannwhitneyu(step_scores, baseline_scores, alternative='two-sided')
stat, p_value_baseline_chain = stats.mannwhitneyu(chain_scores, baseline_scores, alternative='two-sided')
stat, p_value_step_chain = stats.mannwhitneyu(chain_scores, step_scores, alternative='two-sided')
print(f"P-value for Baseline vs. Step-by-Step: {p_value_baseline_step:.3f}")
print(f"P-value for Baseline vs. Chain-of-Thought: {p_value_baseline_chain:.3f}")
print(f"P-value for Step-by-Step vs. Chain-of-Thought: {p_value_step_chain:.3f}")
openai_p_values =[p_value_baseline_step,p_value_baseline_chain,p_value_step_chain]

data = {
    "Comparison": [
        "Baseline vs. Step-by-Step", "Baseline vs. Chain-of-Thought", "Step-by-Step vs. Chain-of-Thought",
        "Baseline vs. Step-by-Step", "Baseline vs. Chain-of-Thought", "Step-by-Step vs. Chain-of-Thought",
        "Baseline vs. Step-by-Step", "Baseline vs. Chain-of-Thought", "Step-by-Step vs. Chain-of-Thought",
        "Baseline vs. Step-by-Step", "Baseline vs. Chain-of-Thought", "Step-by-Step vs. Chain-of-Thought",
        "Baseline vs. Step-by-Step", "Baseline vs. Chain-of-Thought", "Step-by-Step vs. Chain-of-Thought"

    ],
    "Metric": [
        "OpenAI", "OpenAI", "OpenAI",
        "Google-BLEURT", "Google-BLEURT",   "Google-BLEURT",
        "ROUGE","ROUGE","ROUGE",
        "WER","WER","WER",
        "BLEU","BLEU","BLEU"
        
    ],
    "P-Value": [
        openai_p_values[0],openai_p_values[1],openai_p_values[2],
        google_p_values[0],google_p_values[1],google_p_values[2],
        rouge_p_values[0], rouge_p_values[1],rouge_p_values[2],
        wer_p_values[0], wer_p_values[1], wer_p_values[2],
        bleu_values[0],bleu_values[1],bleu_values[2]
    ]
}

df = pd.DataFrame(data)
print(df)
sns.set(style="whitegrid")
plt.figure(figsize=(12, 6))
plot = sns.barplot(x='Comparison', y='P-Value', hue='Metric', data=df, palette='coolwarm')
plt.axhline(0.05, color='red', linewidth=2, linestyle='--')
plot.text(2.5, 0.15, 'Significance threshold (p=0.05)', color = 'red', va='center', ha='center',fontsize=17)
plt.title('P-Values for WER and ROUGE by Comparison Method', fontsize=16)
plt.ylabel('P-Value', fontsize=14)
plt.xlabel('Comparison Methods', fontsize=14)
plt.xticks(rotation=45)  # Rotate x-labels for better readability
plt.tight_layout()
plt.show()

