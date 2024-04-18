import polars as pl
import pandas as pd

def save_data(pl_data, file_name: str, format:str):
    match format:
        case '.xlsx':
            save_excel(pl_data, file_name)
        case '.csv':
            save_csv(pl_data)

def save_excel(pl_data, file_name: str):
    pd_data = pl_data.to_pandas()

    with pd.ExcelWriter(f'./output/{file_name}.xlsx', engine='openpyxl', mode='w') as writer:
        pd_data.to_excel(writer, sheet_name='Sheet1')

def save_csv(pl_data, file_name: str):
    pl_data.write_csv(file_name)