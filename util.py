import numpy as np
import pickle
import gzip
import h5py

def save_zipped_pickle(obj, filename, protocol=-1):
    with gzip.open(filename, 'wb') as f:
        pickle.dump(obj, f, protocol)

def load_zipped_pickle(filename):
    with gzip.open(filename, 'rb') as f:
        loaded_object = pickle.load(f)
        return loaded_object

def save_to_hdf5_file(datasets:dict, file_name:str, compression=None):
    h = h5py.File(file_name, 'w')
    for k, v in datasets.items():
        h.create_dataset(k, data=v, compression=compression)
    h.flush()
    h.close()

def read_from_hdf5_file(file_name:str, ds_name:str):
    h = h5py.File(file_name, 'r')
    return h[ds_name]

def create_record(element_type, value=None):
    if value is None:
        return np.rec.array(np.zeros((1), dtype=element_type)[0], dtype=element_type)
    else:
        return np.rec.array(value, dtype=element_type)
