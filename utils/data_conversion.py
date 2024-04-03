import os
import pickle
import polars as pl

def convert_csv_to_pkl(input_directory:str, input_filename:str, output_directory:str|None=None, output_filename:str|None=None) -> int:
    try:
        pl_data = pl.read_csv(os.path.join(input_directory, input_filename+'.csv'))

        if not output_directory:
            output_directory = input_directory
        if not output_filename:
            output_filename = input_filename
        
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
       
        with open(os.path.join(output_directory, output_filename+'.pkl'), 'wb') as handle:
            pickle.dump(pl_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    except:
        return -1