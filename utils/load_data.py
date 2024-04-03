import pickle

def load_data(full_filename: str):
    file_name = full_filename
    file_extension = '.pkl'

    match file_extension:
        case '.pkl':
            return load_pickle(full_filename)
            
        case _:
            print('Unsupported file type')
            return -1

def load_pickle(full_filename: str):
    with open(full_filename, 'rb') as handle:
        return pickle.load(handle)