import pickle
import polars as pl
from utils.data_conversion import convert_csv_to_pkl
from utils.load_data import load_data
from utils.dataframe_operations import *
from utils.prompt_language_model import *

pl_data = pl.read_csv('./ko_randomly_selected_50.csv')
pl_data = pl_data.with_columns(
    mistral_translation = pl.col('example').map_elements(prompt_mistral)
)

pl_data.write_csv('initial_output.csv')

# print(pl_data)