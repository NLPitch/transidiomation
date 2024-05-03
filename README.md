# Transidiomation
Transidiomation is a prompting framework designed to improve quality of translation on sentences embedded with an idiomatic expression. Further details on the project can be found [here](link to pdf).

## Install
Tranidiomation requires Python version >= 3.10.0. Packages required to use the prompting framework can be installed with the following lines:
```
git clone git@github.com:NLPitch/transidiomation.git
cd transidiomation
git install -r requirements.txt
```

Transidiomation requires you to have an OpenAI API key. If you do not have one, please sign up at [OpenAI](https://openai.com/index/openai-api) and to obtain one.

## How to Run
### Running a Sentence
If you are willing to test out transidiomation on a single sentence, please use the following line:
```
python3 transidiomation.py  --target_language TARGET_LANGUAGE --open_ai_key YOUR_OPEN_AI_KEY --sentence 'SENTENCE_IN_SOURCE_LANGUAGE'
```

### Running a File with Multiple Sentences
If you are willing to test out transidiomation on a batch of sentences, please have a CSV or XLSX file prepared and use the following line:
```
python3 transidiomation.py  --target_language TARGET_LANGUAGE --open_ai_key YOUR_OPEN_AI_KEY --file_path /PATH/TO/SENTENCE/FILE --column_name COLUMN_HEADER [-o /PATH/TO/STORE/OUTPUT]
```
If an output path is not given, the output will by default be stored as `./output.xlsx`

### Output with Baseline
If you would like to compare the output of transidiomation against the two baseline approaches (i.e., result from Google Translate and Naive GPT prompting) add the `-b` flag as follows:
```
python3 transidiomation.py  --target_language TARGET_LANGUAGE --open_ai_key YOUR_OPEN_AI_KEY --sentence SENTENCE_IN_SOURCE_LANGUAGE -b
```