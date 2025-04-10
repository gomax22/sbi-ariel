import argparse
import yaml
import os

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from utils.fm import ariel_resolution, initialise_forward_model, compute_ariel_spectrum
from make_dataset import MJUP, RSOL
import pandas as pd
import random
from utils.prior import default_prior_bounds, restrict_to_prior

import concurrent
import gc

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
random.seed(SEED)



def proxy_compute_ariel_spectrum(opacity_path, cia_path, r_idx, posterior_samples_batch, rs, mp, ariel_wngrid, ariel_wnwidth):
    fm = initialise_forward_model(opacity_path, cia_path)
    spectrum = compute_ariel_spectrum(r_idx, posterior_samples_batch, fm, rs, mp, ariel_wngrid, ariel_wnwidth)
    return spectrum, r_idx

def compute_ariel_spectra(
    eval_dir: str, 
    posterior_samples: np.ndarray, 
    test_mapping: np.ndarray,
    bounds_matrix: np.ndarray,
    aux_data: pd.DataFrame, 
    opacity_path: str, 
    cia_path: str, 
    ):

    n_samples, n_repeats, _ = posterior_samples.shape

    # read in spectral grid
    ariel_wlgrid, ariel_wlwidth, ariel_wngrid, ariel_wnwidth = ariel_resolution()
    
    # ensure the dimensionality matches forward model's input.
    Rs = aux_data['star_radius_m'] / RSOL
    
    # Rp = aux_df['planet_radius_m']/RJUP
    Mp = aux_data['planet_mass_kg'] / MJUP

    posterior_spectra = np.zeros((n_samples, n_repeats, len(ariel_wngrid)))

    spectra_output_dir = os.path.join(eval_dir, "evaluation", "coverage", "spectra")
    if not Path(spectra_output_dir).exists():
        Path(spectra_output_dir).mkdir(exist_ok=True, parents=True)

    # they should be aligned. If not, we have a problem
    with tqdm(total=n_samples * n_repeats, desc="Computing spectra using posterior parameters...") as pbar:
        # for idx, (posterior_samples_batch, pl_idx) in enumerate(zip(posterior_samples, test_mapping)):

        #     rs = Rs[pl_idx]
        #     mp = Mp[pl_idx]

        #     # restrict to prior bounds
        #     posterior_samples_batch = restrict_to_prior(posterior_samples_batch, bounds_matrix)
            
        #     ## compute the spectra
        #     with concurrent.futures.ProcessPoolExecutor(max_workers=16) as pool:
        #         futures = {pool.submit(
        #             proxy_compute_ariel_spectrum,
        #             opacity_path,
        #             cia_path,
        #             r_idx,
        #             posterior_samples_batch,
        #             rs,
        #             mp,
        #             ariel_wngrid,
        #             ariel_wnwidth
        #         ): r_idx for r_idx in range(n_repeats)}

        #         for future in concurrent.futures.as_completed(futures):
        #             pbar.update(1)
        #             spectrum, r_idx = future.result()
        #             posterior_spectra[idx, r_idx] = np.array(spectrum, copy=True)
        #             del futures[future], spectrum
        #             gc.collect()


        rs = Rs[test_mapping[0]]
        mp = Mp[test_mapping[0]]

        # restrict to prior bounds
        posterior_samples_batch = restrict_to_prior(posterior_samples[0], bounds_matrix)
        
        ## compute the spectra
        with concurrent.futures.ProcessPoolExecutor(max_workers=8) as pool:
            futures = {pool.submit(
                proxy_compute_ariel_spectrum,
                opacity_path,
                cia_path,
                r_idx,
                posterior_samples_batch,
                rs,
                mp,
                ariel_wngrid,
                ariel_wnwidth
            ): r_idx for r_idx in range(n_repeats)}

            for future in concurrent.futures.as_completed(futures):
                pbar.update(1)
                spectrum, r_idx = future.result()
                posterior_spectra[0, r_idx] = np.array(spectrum, copy=True)
                del futures[future], spectrum
                gc.collect() 

                
            # for r_idx in range(n_repeats):
            #     pbar.set_description(f"Computing spectra using posterior parameters... ({idx+1}/{n_samples} ({r_idx+1}/{n_repeats}))")

            #     # Initialise base T3 model for ADC2023
            #     # In theory, this may be done once at the init phase.
            #     # To avoid the simulator errors, we init the FM at each observation
            #     fm = initialise_forward_model(opacity_path, cia_path)
            #     spectrum = compute_ariel_spectrum(r_idx, posterior_samples_batch, fm, rs, mp, ariel_wngrid, ariel_wnwidth)
            #     posterior_spectra[idx, r_idx] = spectrum
            #     pbar.update(1)

    # save the spectra
    np.save(os.path.join(spectra_output_dir, 'posterior_spectra.npy'), posterior_spectra)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir", required=True, help="Base save directory for the evaluation")
    parser.add_argument("--settings_file", required=True, help="Settings file for the evaluation")
    args = parser.parse_args()

    # NOT ALL THE ESTIMATORS HAVE SETTINGS (e.g. competitors)
    with open(args.settings_file, "r") as f:
        settings = yaml.safe_load(f)

    # compare models
    # get data path
    data_path = settings["dataset"]["path"]
    test_mapping = np.load(os.path.join(data_path, 'test_mapping.npy'))

    # load auxillary data for forward model
    aux_data = pd.read_csv(settings["adc"]["aux_data"])
    bounds_matrix = default_prior_bounds()
    
    # load posteriors 
    posterior_samples = np.load(os.path.join(args.eval_dir, 'posterior_distribution.npy')) # original domain, (n_samples, n_repeats, n_targets)
    print(f"Loaded posterior samples of shape: {posterior_samples.shape}")

    # compute posterior score 
    compute_ariel_spectra(
        args.eval_dir,
        posterior_samples, 
        test_mapping,
        bounds_matrix,
        aux_data, 
        settings["adc"]["opacity_path"], 
        settings["adc"]["cia_path"], 
    )