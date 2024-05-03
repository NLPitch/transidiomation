import os
import logging
import argparse
import configparser

import polars as pl
import polars.selectors as cs

from utils.load_data import load_data
from utils.data_conversion import *
from utils.save_data import save_data
from utils.translation_methods import *
# from utils.evaluate_translation import *

def main(args):
    target_language = args.target_language.lower()
    open_ai_key = args.open_ai_key 
    bool_run_baseline = args.run_baseline

    if bool_run_baseline:
        pl_iso_codes = load_data('./rsrc/language_iso_codes.csv')
        dict_iso_codes = convert_csv_to_dictionary(pl_iso_codes, 'language', 'iso_code')

        try:
            language_iso = dict_iso_codes[target_language]
        except:
            language_iso = input('ISO code cannot be identified. Please input the ISO language code: ')

    if args.sentence:
        source_text = args.sentence
        print(f"Input Sentence: {source_text}")
        print(f"Transidiomation: {transidiomation(input_sentence=source_text, target_language=target_language, api_key=open_ai_key)}")

        if bool_run_baseline:
            print(f"Google Translate: {baseline_google(input_sentence=source_text, target_language_iso=language_iso)}")
            print(f"Naive GPT: {naiveGPT(input_sentence=source_text, target_language=target_language, api_key=open_ai_key)}")

        return 0

    if args.file_path and args.column_name:
        file_path = args.file_path
        column_name = args.column_name
        try:
            output_path, output_format = os.path.splitext(args.output_path)
        except:
            output_path = './output'
            output_format = '.xlsx'

        pl_data = load_data(file_path)
        pl_data = pl_data.with_columns(
            Transidiomation = pl.col(column_name).map_elements(lambda x: transidiomation(input_sentence=x, target_language=target_language, api_key=open_ai_key))
        )

        if bool_run_baseline:
            pl_data = pl_data.with_columns(
                pl.col(column_name).map_elements(lambda x: baseline_google(input_sentence=x, target_language_iso=language_iso)).alias('Google Translate'),
                pl.col(column_name).map_elements(lambda y: naiveGPT(input_sentence=y, target_language=target_language, api_key=open_ai_key)).alias('Naive GPT'),
            )

        save_data(pl_data, output_path, output_format)
        return 0

    print("INFO: Program ending due to missing source text. Either input a sentence or a CSV file and related column header to run transidiomation.")
    return -1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Translate sentence or sentences based on Transidiomation framework')

    parser.add_argument('--target_language', required=True,
                        help='Target language to translate the sentence into',)
                        
    parser.add_argument('--open_ai_key', required=True,
                        help='Open AI API Key',)

    parser.add_argument('--sentence',
                        help='sentence to be translated')
    parser.add_argument('--file_path',
                        help='path to CSV to translate')
    parser.add_argument('--column_name',
                        help='column header of source language text on the CSV file')

    parser.add_argument('--output_path', '-o',
                        help='path to store the translation output')
    
    parser.add_argument('--run_baseline', '-b', action='store_true',
                        help='add translation result from Google Translate and Naive GPT')

    args = parser.parse_args()
    main(args)