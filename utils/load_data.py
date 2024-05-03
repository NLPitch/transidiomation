import os
import pickle
import polars as pl

def load_data(full_filename: str):
    file_name, file_extension = os.path.splitext(full_filename)

    match file_extension:
        case '.pkl':
            return load_pickle(full_filename)
            
        case '.csv':
            return load_csv(full_filename)
            
        case '.xlsx':
            return load_xlsx(full_filename)

        case '.json':
            return load_json(full_filename)

        case _:
            print('Unsupported file type')
            return -1

def load_pickle(full_filename: str):
    with open(full_filename, 'rb') as handle:
        return pickle.load(handle)
    
def load_csv(full_filename: str):
    return pl.read_csv(full_filename)

def load_xlsx(full_filename: str):
    return pl.read_excel(full_filename)

def load_json(full_filename: str):
    return pl.read_json(full_filename)