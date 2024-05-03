import polars as pl
import pandas as pd

def save_data(pl_data, file_path: str, format:str):
    match format:
        case '.xlsx':
            save_excel(pl_data, file_path)
        case '.csv':
            save_csv(pl_data, file_path)

def save_excel(pl_data, file_path: str):
    pd_data = pl_data.to_pandas()

    with pd.ExcelWriter(f'{file_path}.xlsx', engine='openpyxl', mode='w') as writer:
        pd_data.to_excel(writer, sheet_name='Sheet1')

def save_csv(pl_data, file_path: str):
    pl_data.write_csv(f'{file_path}.csv')