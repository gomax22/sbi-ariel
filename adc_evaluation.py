import argparse
import yaml
import os
import json

import torch
import numpy as np

from pathlib import Path
from tqdm import tqdm
from metrics.adc import compute_spectral_loss
from utils.fm import ariel_resolution, initialise_forward_model, setup_dedicated_fm
from make_dataset import MJUP, RSOL
import pandas as pd
import random
from metrics.adc import compute_posterior_loss
from utils.prior import default_prior_bounds, restrict_to_prior
from plotter.corners import corner_plot_single_distribution
from plotter.spectra import plot_median_spectrum_with_confidence_intervals
from typing import Dict

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
random.seed(SEED)


def compute_ariel_scores(
    eval_dir: str, 
    posterior_samples: np.ndarray, 
    reference_samples: np.ndarray,
    test_loader: torch.utils.data.DataLoader,
    model_label: str,
    bounds_matrix: np.ndarray,
    aux_data: pd.DataFrame, 
    opacity_path: str, 
    cia_path: str, 
    beta: float = 0.8,
    n_quantiles: int = 10
    ):

    n_samples, n_repeats, n_targets = posterior_samples.shape
    wl_channels = next(iter(test_loader))[0].shape[-1]
    weights_distribution = np.ones((posterior_samples.shape[0], posterior_samples.shape[1])) / np.sum(np.ones(posterior_samples.shape[1]))

    labels = [r"$R_p$", r"$T_p$", r"$\log H_2O$",r"$\log CO_2$", r"$\log CO$",r"$\log CH_4$", r"$\log NH_3$"]
    q_list = np.linspace(0.01, 0.99, n_quantiles) ## number of quantiles to sample (fixed to 10 in the competition)

    # read in spectral grid
    ariel_wlgrid, ariel_wlwidth, ariel_wngrid, ariel_wnwidth = ariel_resolution()
    
    # ensure the dimensionality matches forward model's input.
    Rs = aux_data['star_radius_m'] / RSOL
    
    # Rp = aux_df['planet_radius_m']/RJUP
    Mp = aux_data['planet_mass_kg'] / MJUP

    spectral_scores = []
    posterior_scores = np.zeros((n_samples + 1, n_targets + 1))

    posterior_median_spectra = []
    posterior_bound_spectra = []
    reference_median_spectra = []
    reference_bound_spectra = []

    
    posterior_real_spectrum_loss = np.zeros((n_samples + 1, wl_channels + 1))
    posterior_ideal_spectrum_loss = np.zeros((n_samples + 1, wl_channels + 1))
    posterior_theta_loss = np.zeros((n_samples + 1, n_targets + 1))

    reference_real_spectrum_loss = np.zeros((n_samples + 1, wl_channels + 1))
    reference_ideal_spectrum_loss = np.zeros((n_samples + 1, wl_channels + 1))
    reference_theta_loss = np.zeros((n_samples + 1, n_targets + 1))

    adc_output_dir = os.path.join(eval_dir, "evaluation", "adc")

    if not Path(adc_output_dir).exists():
        Path(adc_output_dir).mkdir(exist_ok=True, parents=True)

    # they should be aligned. If not, we have a problem
    with tqdm(total=n_samples * 3, desc="Computing scores...") as pbar:
        for idx, (posterior_samples_batch, reference_samples_batch, (ideal_spectrum, noise, real_spectrum, theta, pl_idx)) in enumerate(zip(posterior_samples, reference_samples, test_loader)):
            weights = weights_distribution[idx]

            theta = theta.numpy().reshape(-1)
            ideal_spectrum = ideal_spectrum.numpy().reshape(-1)
            real_spectrum = real_spectrum.numpy().reshape(-1)
            noise = noise.numpy().reshape(-1)

            trace_output_dir = os.path.join(adc_output_dir, f"trace_{idx}")
            if not Path(trace_output_dir).exists():
                Path(trace_output_dir).mkdir(exist_ok=True, parents=True)
    
            pbar.set_description(f"Computing posterior score... ({idx+1}/{n_samples})")

            # restrict to prior bounds
            posterior_samples_batch = restrict_to_prior(posterior_samples_batch, bounds_matrix)
            reference_samples_batch = restrict_to_prior(reference_samples_batch, bounds_matrix)

            ## posterior score
            p_scores = compute_posterior_loss(
                posterior_samples_batch,
                reference_samples_batch,
                bounds_matrix,
                restrict=True, # already done in the previous step
                normalize=True
            )
            pbar.update(1)

            corner_plot_single_distribution(
                posterior_samples_batch,
                theta,
                labels,
                model_label,
                os.path.join(trace_output_dir, f"corner_{idx}.png")

            )

            pbar.set_description(f"Setting up dedicated FM... ({idx+1}/{n_samples})")
            
            # Initialise base T3 model for ADC2023
            # In theory, this may be done once at the init phase.
            # To avoid the simulator errors, we init the FM at each observation
            fm = initialise_forward_model(opacity_path, cia_path)
            
            ## spectral score
            proxy_compute_spectrum = setup_dedicated_fm(fm, idx, Rs, Mp, ariel_wngrid, ariel_wnwidth)
            pbar.update(1)

            pbar.set_description(f"Computing spectral score... ({idx+1}/{n_samples})")
            s_score, posterior_median, posterior_bound, reference_median, reference_bound = compute_spectral_loss(
                posterior_samples_batch, 
                reference_samples_batch, 
                weights, 
                bounds_matrix, 
                proxy_compute_spectrum, 
                q_list
            )
            print(f"Posterior score: {p_scores.mean():.4f}, Spectral score: {s_score:.4f}")
            posterior_scores[idx, :-1] = p_scores
            spectral_scores.append(s_score)

            # store median and bounds as their computation is expensive
            posterior_median_spectra.append(posterior_median)
            posterior_bound_spectra.append(posterior_bound)

            reference_median_spectra.append(reference_median)
            reference_bound_spectra.append(reference_bound)

            # compute loss
            # compute difference between spectrum and model median spectrum
            posterior_real_spectrum_loss[idx, :wl_channels] = (real_spectrum - posterior_median) ** 2
            reference_real_spectrum_loss[idx, :wl_channels] = (real_spectrum - reference_median) ** 2
            posterior_ideal_spectrum_loss[idx, :wl_channels] = (ideal_spectrum - posterior_median) ** 2
            reference_ideal_spectrum_loss[idx, :wl_channels] = (ideal_spectrum - reference_median) ** 2
            
            # we should consider the median theta for the model and NS
            # a possible selection crition for the median theta may be the one associated to the median log probability (join)
            # otherwise we can consider the median theta for each target parameter but this may be not the best choice
            posterior_theta_loss[idx, :n_targets] = (theta - posterior_samples_batch.mean(axis=0)) ** 2
            reference_theta_loss[idx, :n_targets] = (theta - reference_samples_batch.mean(axis=0)) ** 2

               
            plot_median_spectrum_with_confidence_intervals(
                ariel_wlgrid,
                posterior_median,
                posterior_bound,
                real_spectrum,
                ideal_spectrum,
                noise,
                model_label,
                os.path.join(trace_output_dir, f"spectrum_{idx}.png")
            )

            pbar.update(1)

    # store the median and bounds
    posterior_median_spectra = np.array(posterior_median_spectra)
    posterior_bound_spectra = np.array(posterior_bound_spectra)

    reference_median_spectra = np.array(reference_median_spectra)
    reference_bound_spectra = np.array(reference_bound_spectra)

    median_spectra_output_dir = os.path.join(adc_output_dir, "median_spectra")
    if not Path(median_spectra_output_dir).exists():
        Path(median_spectra_output_dir).mkdir(exist_ok=True, parents=True)

    np.save(os.path.join(median_spectra_output_dir, 'posterior_median_spectra.npy'), posterior_median_spectra)
    np.save(os.path.join(median_spectra_output_dir, 'reference_median_spectra.npy'), reference_median_spectra)


    bounds_spectra_output_dir = os.path.join(adc_output_dir, "bounds_spectra")
    if not Path(bounds_spectra_output_dir).exists():
        Path(bounds_spectra_output_dir).mkdir(exist_ok=True, parents=True)

    np.save(os.path.join(bounds_spectra_output_dir, 'posterior_bound_spectra.npy'), posterior_bound_spectra)
    np.save(os.path.join(bounds_spectra_output_dir, 'reference_bound_spectra.npy'), reference_bound_spectra)


    # complete the last row and the last column with the mean values for model
    posterior_real_spectrum_loss[-1, :wl_channels] = np.mean(posterior_real_spectrum_loss[:-1, :wl_channels], axis=0)
    posterior_real_spectrum_loss[:-1, wl_channels] = np.mean(posterior_real_spectrum_loss[:-1, :wl_channels], axis=1)
    posterior_real_spectrum_loss[-1, -1] = np.mean(posterior_real_spectrum_loss[:-1, :wl_channels])
    avg_posterior_real_spectrum_loss = posterior_real_spectrum_loss[-1, -1]

    posterior_ideal_spectrum_loss[-1, :wl_channels] = np.mean(posterior_ideal_spectrum_loss[:-1, :wl_channels], axis=0)
    posterior_ideal_spectrum_loss[:-1, wl_channels] = np.mean(posterior_ideal_spectrum_loss[:-1, :wl_channels], axis=1)
    posterior_ideal_spectrum_loss[-1, -1] = np.mean(posterior_ideal_spectrum_loss[:-1, :wl_channels])
    avg_posterior_ideal_spectrum_loss = posterior_ideal_spectrum_loss[-1, -1]

    reference_real_spectrum_loss[-1, :wl_channels] = np.mean(reference_real_spectrum_loss[:-1, :wl_channels], axis=0)
    reference_real_spectrum_loss[:-1, wl_channels] = np.mean(reference_real_spectrum_loss[:-1, :wl_channels], axis=1)
    reference_real_spectrum_loss[-1, -1] = np.mean(reference_real_spectrum_loss[:-1, :wl_channels])
    avg_reference_real_spectrum_loss = reference_real_spectrum_loss[-1, -1]


    reference_ideal_spectrum_loss[-1, :wl_channels] = np.mean(reference_ideal_spectrum_loss[:-1, :wl_channels], axis=0)
    reference_ideal_spectrum_loss[:-1, wl_channels] = np.mean(reference_ideal_spectrum_loss[:-1, :wl_channels], axis=1)
    reference_ideal_spectrum_loss[-1, -1] = np.mean(reference_ideal_spectrum_loss[:-1, :wl_channels])
    avg_reference_ideal_spectrum_loss = reference_ideal_spectrum_loss[-1, -1]

    posterior_theta_loss[-1, :n_targets] = np.mean(posterior_theta_loss[:-1, :n_targets], axis=0)
    posterior_theta_loss[:-1, n_targets] = np.mean(posterior_theta_loss[:-1, :n_targets], axis=1)
    posterior_theta_loss[-1, -1] = np.mean(posterior_theta_loss[:-1, :n_targets])
    avg_posterior_theta_loss = posterior_theta_loss[-1, -1]

    reference_theta_loss[-1, :n_targets] = np.mean(reference_theta_loss[:-1, :n_targets], axis=0)
    reference_theta_loss[:-1, n_targets] = np.mean(reference_theta_loss[:-1, :n_targets], axis=1)
    reference_theta_loss[-1, -1] = np.mean(reference_theta_loss[:-1, :n_targets])
    avg_reference_theta_loss = reference_theta_loss[-1, -1]

    # save spectrum and theta loss
    losses_output_dir = os.path.join(adc_output_dir, "losses")
    if not Path(losses_output_dir).exists():
        Path(losses_output_dir).mkdir(exist_ok=True, parents=True)
    
    np.save(os.path.join(losses_output_dir, 'posterior_real_spectrum_loss.npy'), posterior_real_spectrum_loss)
    np.save(os.path.join(losses_output_dir, 'posterior_ideal_spectrum_loss.npy'), posterior_ideal_spectrum_loss)
    np.save(os.path.join(losses_output_dir, 'reference_real_spectrum_loss.npy'), reference_real_spectrum_loss)
    np.save(os.path.join(losses_output_dir, 'reference_ideal_spectrum_loss.npy'), reference_ideal_spectrum_loss)
    np.save(os.path.join(losses_output_dir, 'posterior_theta_loss.npy'), posterior_theta_loss)
    np.save(os.path.join(losses_output_dir, 'reference_theta_loss.npy'), reference_theta_loss)
    

    # make a unique dataframe for posterior and spectral scores with final score
    # compute the average scores
    posterior_scores[-1, :-1] = np.mean(posterior_scores[:-1, :-1], axis=0)
    posterior_scores[:-1, -1] = np.mean(posterior_scores[:-1, :-1], axis=1)
    posterior_scores[-1, -1] = np.mean(posterior_scores[:-1, :-1])
    avg_posterior_score = posterior_scores[-1, -1]

    spectral_scores = np.array(spectral_scores)
    avg_spectral_score = np.mean(spectral_scores)
    spectral_scores = np.concatenate([spectral_scores, avg_spectral_score.reshape(1)], axis=0)

    # save the scores
    scores_output_dir = os.path.join(adc_output_dir, "scores")
    if not Path(scores_output_dir).exists():
        Path(scores_output_dir).mkdir(exist_ok=True, parents=True)
    
    np.save(os.path.join(scores_output_dir, 'posterior_scores.npy'), posterior_scores)
    np.save(os.path.join(scores_output_dir, 'spectral_scores.npy'), spectral_scores)
    
    posterior_scores_df = pd.DataFrame(posterior_scores, columns=labels + ['avg_posterior_score'])
    spectral_scores_df = pd.DataFrame(spectral_scores, columns=['avg_spectral_score'])

    posterior_scores_df.to_csv(os.path.join(scores_output_dir, 'posterior_scores.csv'), index=False)
    spectral_scores_df.to_csv(os.path.join(scores_output_dir, 'spectral_scores.csv'), index=False)

    # compute final score
    final_score = (1 - beta) * avg_spectral_score + beta * avg_posterior_score

    adc_results = {
        "avg_posterior_score": avg_posterior_score.item(),
        "avg_spectral_score": avg_spectral_score.item(),
        "final_score": final_score.item(),
    }

    losses_results = {
        "avg_posterior_real_spectrum_loss": avg_posterior_real_spectrum_loss.item(),
        "avg_posterior_ideal_spectrum_loss": avg_posterior_ideal_spectrum_loss.item(),
        "avg_reference_real_spectrum_loss": avg_reference_real_spectrum_loss.item(),
        "avg_reference_ideal_spectrum_loss": avg_reference_ideal_spectrum_loss.item(),
        "avg_posterior_theta_loss": avg_posterior_theta_loss.item(),
        "avg_reference_theta_loss": avg_reference_theta_loss.item(),
    }

    with open(os.path.join(adc_output_dir, "results.json"), "w") as f:
        json.dump(adc_results, f)


    with open(os.path.join(losses_output_dir, "results.json"), "w") as f:
        json.dump(losses_results, f)

    return adc_results, losses_results

    


def perform_adc_evaluation(
    eval_dir: str, 
    settings: Dict, 
    model_label: str,
    ns_dir: str
):

    # get data path
    data_path = settings["dataset"]["path"]

    # load auxillary data for forward model
    aux_data = pd.read_csv(settings["adc"]["aux_data"])
    bounds_matrix = default_prior_bounds()
    
    # load posteriors 
    posterior_samples = np.load(os.path.join(eval_dir, 'posterior_distribution.npy')) # original domain, (n_samples, n_repeats, n_targets)
    reference_samples = np.load(os.path.join(ns_dir, 'posterior_distribution.npy'))

    # load thetas
    test_real_spectra = torch.tensor(np.load(os.path.join(data_path, 'test_real_spectra.npy')))
    test_ideal_spectra = torch.tensor(np.load(os.path.join(data_path, 'test_ideal_spectra.npy')))
    test_targets = torch.tensor(np.load(os.path.join(data_path, 'test_targets.npy')))
    test_noises = torch.tensor(np.load(os.path.join(data_path, 'test_noises.npy')))
    test_mapping = torch.tensor(np.load(os.path.join(data_path, 'test_mapping.npy')))
    
    # build test dataset
    test_dataset = torch.utils.data.TensorDataset(
        test_ideal_spectra, test_noises, test_real_spectra, test_targets, test_mapping
    )    

    # build data loader        
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
    )

    # compute posterior score 
    adc_results, losses_results = compute_ariel_scores(
        eval_dir,
        posterior_samples, 
        reference_samples,
        test_loader,
        model_label,
        bounds_matrix,
        aux_data, 
        settings["adc"]["opacity_path"], 
        settings["adc"]["cia_path"], 
        settings["adc"]["beta"],
        settings["adc"]["n_quantiles"]
    )
    
    return adc_results, losses_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir", required=True, help="Base save directory for the evaluation")
    parser.add_argument("--ns_dir", required=True, help="Base save directory for Nested Sampling")
    parser.add_argument("--settings_file", required=True, help="Settings file for the evaluation")
    parser.add_argument("--model_label", required=True, help="Model label for plotting and saving results")
    args = parser.parse_args()

    # NOT ALL THE ESTIMATORS HAVE SETTINGS (e.g. competitors)
    with open(args.settings_file, "r") as f:
        settings = yaml.safe_load(f)

    # compare models
    adc_results, losses_results = perform_adc_evaluation(
        eval_dir=args.eval_dir, 
        settings=settings, 
        model_label=str(args.model_label),
        ns_dir=args.ns_dir,
    )

    print("ADC Evaluation results:")
    for key, value in adc_results.items():
        print(f"{key}: {value}")

    for key, value in losses_results.items():
        print(f"{key}: {value}")


    ##### EVALUATION ON TEST SPLIT
    # update settings.yaml with evaluation results
    # with open(os.path.join(args.eval_dir, "settings.yaml"), "w") as f:
    #     yaml.dump(settings, f)