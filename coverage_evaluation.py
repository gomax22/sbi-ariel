import argparse
import yaml
import os
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
import random
from typing import Dict
from metrics.coverage import coverage_analysis
from pathlib import Path

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
random.seed(SEED)


def compute_coverage_analysis(
    eval_dir: str, 
    posterior_samples: np.ndarray, 
    thetas: np.ndarray
    ):


    n_samples = posterior_samples.shape[0]
    
    # 5 ==> posterior_mean, posterior_std, posterior_1sigma, posterior_2sigma, posterior_support, 2 * 3 ==> ratio, all for 1_sigma, 2_sigma, support
    # posterior_evaluation = np.zeros((n_samples, 5 * posterior_samples.shape[-1] + 2 * 3))
    
    columns_label = ['posterior_mean_R_p', 'posterior_mean_T_p', 'posterior_mean_log_H2O', 'posterior_mean_log_CO2', 'posterior_mean_log_CO', 'posterior_mean_log_CH4', 'posterior_mean_log_NH3', 
                     'posterior_std_R_p', 'posterior_std_T_p', 'posterior_std_log_H2O', 'posterior_std_log_CO2', 'posterior_std_log_CO', 'posterior_std_log_CH4', 'posterior_std_log_NH3',
                     'posterior_accuracy_R_p_one_sigma', 'posterior_accuracy_T_p_one_sigma', 'posterior_accuracy_log_H2O_one_sigma', 'posterior_accuracy_log_CO2_one_sigma', 'posterior_accuracy_log_CO_one_sigma', 'posterior_accuracy_log_CH4_one_sigma', 'posterior_accuracy_log_NH3_one_sigma', 'posterior_accuracy_ratio_one_sigma', 'posterior_accuracy_all_one_sigma',
                     'posterior_accuracy_R_p_two_sigma', 'posterior_accuracy_T_p_two_sigma', 'posterior_accuracy_log_H2O_two_sigma', 'posterior_accuracy_log_CO2_two_sigma', 'posterior_accuracy_log_CO_two_sigma', 'posterior_accuracy_log_CH4_two_sigma', 'posterior_accuracy_log_NH3_two_sigma', 'posterior_accuracy_ratio_two_sigma', 'posterior_accuracy_all_two_sigma',
                     'posterior_accuracy_R_p_support', 'posterior_accuracy_T_p_support', 'posterior_accuracy_log_H2O_support', 'posterior_accuracy_log_CO2_support', 'posterior_accuracy_log_CO_support', 'posterior_accuracy_log_CH4_support', 'posterior_accuracy_log_NH3_support', 'posterior_accuracy_ratio_support', 'posterior_accuracy_all_support'
    ]


    posterior_means =   []
    posterior_stds  =   []
    coverage_1sigma      =   []
    coverage_2sigma      =   []
    coverage_support     =   []

    # they should be aligned. If not, we have a problem
    with tqdm(total=n_samples, desc="Computing prior-predictive checks...") as pbar:
        for idx, (posterior_samples_batch, theta) in enumerate(zip(posterior_samples, thetas)):
            
            # compute means and stds
            posterior_means.append(np.mean(posterior_samples_batch, axis=0))
            posterior_stds.append(np.std(posterior_samples_batch, axis=0, ddof=1))

            # compute coverages
            coverage_results = coverage_analysis(posterior_samples_batch, theta)
            coverage_support.append(
                np.concatenate([
                    coverage_results['support']['independent_accuracy'],
                    coverage_results['support']['avg_accuracy'],
                    coverage_results['support']['joint_accuracy']
                ], axis=0)
            )

            coverage_1sigma.append(
                np.concatenate([
                    coverage_results['one_sigma']['independent_accuracy'],
                    coverage_results['one_sigma']['avg_accuracy'],
                    coverage_results['one_sigma']['joint_accuracy']
                ], axis=0)
            )
            
            coverage_2sigma.append(
                np.concatenate([
                    coverage_results['two_sigma']['independent_accuracy'],
                    coverage_results['two_sigma']['avg_accuracy'],
                    coverage_results['two_sigma']['joint_accuracy']
                ], axis=0)
            )
                        
            pbar.update(1)
    
    posterior_evaluation = np.concatenate([posterior_means, posterior_stds, coverage_1sigma, coverage_2sigma, coverage_support], axis=1)

    output_dir = os.path.join(eval_dir, 'evaluation', 'coverage')
    if not Path(output_dir).exists():
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    # create dataframe starting from posterior evaluation
    posterior_evaluation_df = pd.DataFrame(posterior_evaluation, columns=columns_label)    
    posterior_evaluation_df.to_csv(os.path.join(output_dir, 'evaluation.csv'), index=False)
    # np.save(os.path.join(eval_dir, 'evaluation.npy'), posterior_evaluation)


def perform_coverage_evaluation(
    eval_dir: str, 
    data_dir: str,
    ):
    
    # load test dataset
    test_targets = np.load(os.path.join(data_dir, 'test_targets.npy'))
    
    # load posteriors 
    posterior_samples = np.load(os.path.join(eval_dir, 'posterior_distribution.npy')) # original domain, (n_samples, n_repeats, n_targets)

    compute_coverage_analysis(
        eval_dir,
        posterior_samples, 
        test_targets
    )

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir", required=True, help="Base save directory for the evaluation")
    parser.add_argument("--settings_file", required=True, help="Settings file for the evaluation")
    args = vars(parser.parse_args())

    eval_dir = args['eval_dir']
    settings_file = args['settings_file']

    with open(settings_file, 'r') as f:
        settings = yaml.safe_load(f)
    
    data_dir = settings['dataset']['path']

    # compare models
    perform_coverage_evaluation(
        eval_dir,
        data_dir
    )
