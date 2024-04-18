import pickle
import polars as pl
from utils.data_conversion import convert_csv_to_pkl
from utils.load_data import load_data
from utils.save_data import save_data
from utils.dataframe_operations import *
from utils.prompt_language_model import *

# file_name = '/home/yaoyi/pyo00005/CSci5541/transidiomation/data/xlsx/best_idioms_kiss.xlsx'
file_name = '/home/yaoyi/pyo00005/CSci5541/transidiomation/output/naive_translation.xlsx'

pl_data1 = load_data(file_name).select(
    pl.col(['Idiom', 'Sentence', 'Translation', 'Matching']),
    zero_control = pl.col('open_ai_naive')
)
pl_data2 = load_data('/home/yaoyi/pyo00005/CSci5541/transidiomation/output/zero_step.xlsx').select(
    pl.col(['Idiom', 'Sentence', 'Translation', 'Matching']),
    zero_experiment = pl.col('open_ai_with_step')
)
pl_data3 = load_data('/home/yaoyi/pyo00005/CSci5541/transidiomation/output/one_shot.xlsx').select(
    pl.col(['Idiom', 'Sentence', 'Translation', 'Matching']),
    one_control = pl.col('open_ai_one_shot')
)
pl_data4 = load_data('/home/yaoyi/pyo00005/CSci5541/transidiomation/output/one_step.xlsx').select(
    pl.col(['Idiom', 'Sentence', 'Translation', 'Matching']),
    one_experiment = pl.col('open_ai_one_step')
)
pl_data5 = load_data('/home/yaoyi/pyo00005/CSci5541/transidiomation/output/two_shot.xlsx').select(
    pl.col(['Idiom', 'Sentence', 'Translation', 'Matching']),
    two_control = pl.col('open_ai_one_step')
)
pl_data6 = load_data('/home/yaoyi/pyo00005/CSci5541/transidiomation/output/two_step.xlsx').select(
    pl.col(['Idiom', 'Sentence', 'Translation', 'Matching']),
    two_experiment = pl.col('open_ai_one_step')
)
pl_data7 = load_data('/home/yaoyi/pyo00005/CSci5541/transidiomation/output/five_shot.xlsx').select(
    pl.col(['Idiom', 'Sentence', 'Translation', 'Matching']),
    five_control = pl.col('open_ai_one_step')
)
pl_data8 = load_data('/home/yaoyi/pyo00005/CSci5541/transidiomation/output/five_step.xlsx').select(
    pl.col(['Idiom', 'Sentence', 'Translation', 'Matching']),
    five_experiment = pl.col('open_ai_one_step')
)

pl_data = pl.concat(
    [pl_data1, pl_data2, pl_data3, pl_data4, pl_data5, pl_data6, pl_data7, pl_data8],
    how='align'
)

pl_data = pl_data.select(
    pl.col(['Idiom', 'Sentence', 'Translation', 'Matching']),
    pl.col(['zero_experiment', 'one_experiment', 'two_experiment', 'five_experiment']).str.split('Step 1.').list.get(1)
).with_columns(
    pl.col(['zero_experiment', 'one_experiment', 'two_experiment', 'five_experiment']).str.split('Step 2.').list.get(0)
)

# pl_data = pl_data.with_columns(
#     pl.col(['zero_experiment', 'one_experiment', 'two_experiment', 'five_experiment']).str.split('Step 3.').list.get(1)
# )

# pl_data = load_data('/home/yaoyi/pyo00005/CSci5541/transidiomation/output/one_step.xlsx')

print(pl_data)
print(pl_data.columns)

# pl_data = pl_data.with_columns(
#     open_ai_one_step = pl.col('Sentence').map_elements(lambda x: prompt_openai(x))
# )
# save_data(pl_data, 'all_translation', '.xlsx')
save_data(pl_data, 'all_idiom_identify', '.xlsx')

# input_message = {
#     'zero-shot': [{"role": "user", "content": f"Translate {source_text} to English"}],
#     'zero-step': [],
#     'one-shot': [],
#     'one-step': [],
#     'two-shot': [],
#     'two-step': [],
#     'five-shot': [],
#     'five-step': [],
# }