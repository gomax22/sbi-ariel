import numpy as np
import time
from functools import wraps
import pandas as pd
import json
import pickle


def augment_data_with_noise(spectra, noise, repeat ):
    aug_spectra = augment_data(spectra, noise, repeat)
    aug_spectra = aug_spectra.reshape(-1, spectra.shape[1])
    return aug_spectra

def augment_data(arr, noise, repeat=10):
    noise_profile = np.random.normal(loc=0, scale=noise, size=(repeat,arr.shape[0], arr.shape[1]))
    ## produce noised version of the spectra
    aug_arr = arr[np.newaxis, ...] + noise_profile
    return aug_arr

def to_observed_matrix(data_file, aux_file):
    # careful, orders in data files are scambled. We need to "align them with id from aux file"
    num = len(data_file.keys())
    id_order = aux_file['planet_ID'].to_numpy()
    observed_spectrum = np.zeros((num,52,4)) # hardcoded

    for idx, x in enumerate(id_order):
        current_planet_id = f'Planet_{x}'
        instrument_wlgrid = data_file[current_planet_id]['instrument_wlgrid'][:]
        instrument_spectrum = data_file[current_planet_id]['instrument_spectrum'][:]
        instrument_noise = data_file[current_planet_id]['instrument_noise'][:]
        instrument_wlwidth = data_file[current_planet_id]['instrument_width'][:]
        observed_spectrum[idx,:,:] = np.concatenate([instrument_wlgrid[...,np.newaxis],
                                            instrument_spectrum[...,np.newaxis],
                                            instrument_noise[...,np.newaxis],
                                            instrument_wlwidth[...,np.newaxis]],axis=-1)
    return observed_spectrum


def augment_data_with_noise(spectra, noise, repeat):
    aug_spectra = augment_data(spectra, noise, repeat)
    aug_spectra = aug_spectra.reshape(-1, spectra.shape[1])
    return aug_spectra

def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time

        # format time in hours, minutes, seconds
        hours, rem = divmod(total_time, 3600)
        minutes, seconds = divmod(rem, 60)

        time_str = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)
        print(f'{func.__name__} Took {time_str}')
        return result
    return timeit_wrapper

def read_data(path):
    import h5py
    trace = h5py.File(path,"r")
    return trace


def to_matrix(data_file_gt, predicted_file, aux_df):
    # careful, orders in data files are scambled. We need to "align them with id from aux file"
    id_order = aux_df.index
    num = len(data_file_gt.keys())
    list_trace_gt = []
    list_weight_gt = []
    list_trace_predicted = []
    list_weight_predicted = []

    for x in id_order:
        current_planet_id = f'Planet_{x}'
        trace_gt_planet = np.array(data_file_gt[current_planet_id]['tracedata'])
        trace_weight_planet = np.array(data_file_gt[current_planet_id]['weights'])
        list_trace_gt.append(trace_gt_planet)
        list_weight_gt.append(trace_weight_planet)
        predicted_planet = np.array(predicted_file[current_planet_id]['tracedata'])
        predicted_weights = np.array(predicted_file[current_planet_id]['weights'])
        list_trace_predicted.append(predicted_planet)
        list_weight_predicted.append(predicted_weights)
        
    return list_trace_gt, list_weight_gt, list_trace_predicted,list_weight_predicted


def load_np(fpath: str, **kwargs):
    return np.load(fpath, **kwargs)

def load_json(fpath: str, **kwargs):
    with open(fpath, 'r') as f:
        return json.load(f, **kwargs)
    
def load_csv(fpath: str, **kwargs):
    return pd.read_csv(fpath, **kwargs)

def load_pkl(fpath: str, **kwargs):
    with open(fpath, 'rb') as f:
        return pickle.load(f, **kwargs)

def load_fn(
    ext: str,
):
    if ext == 'npy':
        return load_np
    elif ext == 'pkl':
        return load_pkl
    elif ext == 'json':
        return load_json
    elif ext == 'csv':
        return load_csv
    else:
        raise ValueError(f"Extension {ext} not supported")