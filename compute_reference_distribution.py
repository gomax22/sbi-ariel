import h5py
import argparse
import os
import numpy as np
import torch
from utils.prior import default_prior_bounds, default_prior_bounds_level2data, restrict_to_prior
from utils.resampling import resample_equal  
import pandas as pd
import random
from pathlib import Path
import corner

# fix random seed for reproducibility
SEED = 123
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def compute_reference_distribution_using_resize(ns_traces, n_repeats):
    n_samples = len(ns_traces)
    n_targets = ns_traces[0].shape[-1]
    resized_reference_distribution = np.zeros((n_samples, n_repeats, n_targets))


    for idx, trace in enumerate(ns_traces):
        resampled_traces = []
        for t in range(n_targets):
            resampled_trace = np.resize(trace[:, t], (n_repeats,))
            resampled_traces.append(resampled_trace)
        
        resampled_traces = np.stack(resampled_traces, axis=1)
        resized_reference_distribution[idx] = resampled_traces

    return resized_reference_distribution


# ??
# def compute_reference_distribution_using_random_sampling(ns_traces, posterior_samples):
#     n_test_samples, n_repeats, n_targets = posterior_samples.shape
#     min_repeats = min(np.min([trace.shape[0] for trace in ns_traces]), n_repeats)

#     resized_reference_distribution = np.zeros((n_test_samples, min_repeats, n_targets))
#     resized_posterior_distribution = np.zeros((n_test_samples, min_repeats, n_targets))

#     for idx, (trace, posterior_samples_batch) in enumerate(zip(ns_traces, posterior_samples)):
#         ref_indices = np.random.choice(range(trace.shape[0]), min_repeats, replace=False)
#         post_indices = np.random.choice(range(posterior_samples_batch.shape[0]), min_repeats, replace=False)

#         resized_reference_distribution[idx] = trace[ref_indices]
#         resized_posterior_distribution[idx] = posterior_samples_batch[post_indices]

#     return resized_reference_distribution, resized_posterior_distribution
    

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ground_truth_dir", type=str, required=True, help="Path to the data directory")
    ap.add_argument("--data_dir", type=str, required=True, help="Path to the data directory")
    ap.add_argument("--n_repeats", type=int, default=2048, help="Number of repeats")
    ap.add_argument("--output_dir", type=str, required=True, help="Path to the output directory")
    ap.add_argument("--adc", action="store_true", help="Use ADC specifications data otherwise Level2Data specifications")
    args = ap.parse_args()

    tracedata_file = h5py.File(os.path.join(args.ground_truth_dir, "Tracedata.hdf5"), "r")
    test_mapping = np.load(os.path.join(args.data_dir, "test_mapping.npy"))
   
    if args.adc:
        fm_params = pd.read_csv(os.path.join(args.ground_truth_dir, "FM_Parameter_Table.csv")).drop(columns=["Unnamed: 0"])
        bounds_matrix = default_prior_bounds()
        planetlist = sorted([key for key in tracedata_file.keys() if tracedata_file[key]['weights'].shape])
        planetlist = [key for key in planetlist if int(key[12:]) - 1 in test_mapping]
    else:
        fm_params = pd.read_csv(os.path.join(args.ground_truth_dir, "FM_Parameter_Table.csv"))
        bounds_matrix = default_prior_bounds_level2data()
        planetlist = sorted([key for key in tracedata_file.keys() if tracedata_file[key]['weights'].shape])
        planetlist = [key for key in planetlist if int(key[7:]) in test_mapping]
        
    print(f"Number of planets with ground truth: {len(planetlist)}")

    num_samples = []
    traces = []

    for p_idx, key in enumerate(planetlist):
        trace = tracedata_file[key]['tracedata'][:]
        weights = tracedata_file[key]['weights'][:]
        
        # possible no ground truth in test data 
        if np.isnan(trace).sum() == 1:
            continue

        trace = restrict_to_prior(resample_equal(trace, weights), bounds_matrix)
        traces.append(trace)
        
        num_samples.append(trace.shape[0])

        key_query = key[7:] if args.adc else int(key[7:])
        thetas = fm_params[fm_params['planet_ID'] == key_query].iloc[:, 1:].to_numpy()
        n_targets = trace.shape[-1]

    resized_reference_distribution = compute_reference_distribution_using_resize(traces, args.n_repeats)

    if not Path(args.output_dir).exists():
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Resized reference distribution shape: {resized_reference_distribution.shape}")
    np.save(os.path.join(args.output_dir, "posterior_distribution.npy"), resized_reference_distribution)

# python compute_reference_distribution.py --ground_truth_dir "ARIEL/TrainingData/Ground Truth Package" --data_dir sbi-ariel/data --n_repeats 2048 --output_dir ns_runs --adc