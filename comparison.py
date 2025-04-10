import argparse
import os
import pickle
import pprint
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import yaml
from utils.utils import load_fn
from plotter.corners import (
    corner_plot_multiple_distributions, 
    corner_plot_prior_posteriors,
)
from plotter.spectra import plot_median_spectra_with_confidence_intervals
from plotter.diagrams import plot_regression_diagrams
from utils.fm import ariel_resolution

TASK_TO_PATH = {
    "adc": os.path.join("evaluation", "adc", "results.json"),
    "losses": os.path.join("evaluation", "adc", "losses", "results.json"),
    "coverage": os.path.join("evaluation", "coverage", "evaluation.csv"),
    "posterior": os.path.join("posterior_distribution.npy"),
    "sbc": os.path.join("calibration", "sbc", "stats.pkl"), # TODO: check stats object
    "tarp": os.path.join("calibration", "tarp", "stats.pkl"), # TODO: check stats object
    "median_spectra": os.path.join("evaluation", "adc", "median_spectra", "posterior_median_spectra.npy"),
    "bounds_spectra": os.path.join("evaluation", "adc", "bounds_spectra", "posterior_bound_spectra.npy"),
    "regression": os.path.join("calibration", "regression", "metrics.pkl"),
    "diagrams": os.path.join("calibration", "diagrams")
}


def load_evaluation_results(
    eval_dir: str,
    task: str,
):
    eval_path = TASK_TO_PATH[task]
    results = []

    fpath = os.path.join(eval_dir, eval_path)
        
    ext = fpath.split('.')[-1]
    load_func = load_fn(ext)
    results.append(load_func(fpath))

    return results

if __name__ == "__main__":
    tasks = ['adc', 'losses', 'coverage', 'posterior', 'sbc', 'tarp', 
             'median_spectra', 'bounds_spectra', 'regression']
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings_file", required=True, help="Settings file for the comparrison")
    ap.add_argument("--tasks", nargs='+', default=tasks, help="Tasks to compare")
    ap.add_argument("--output_dir", default='comparison', help="Output directory for the comparison")
    args = ap.parse_args()

    # create output directory
    if not os.path.exists(args.output_dir):
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # load settings file for comparison
    with open(args.settings_file, 'r') as f:
        settings = yaml.safe_load(f)

    # load test dataset
    test_targets = np.load(os.path.join(settings['dataset']['path'], 'test_targets.npy'))
    test_ideal_spectra = np.load(os.path.join(settings['dataset']['path'], 'test_ideal_spectra.npy'))
    test_real_spectra = np.load(os.path.join(settings['dataset']['path'], 'test_real_spectra.npy'))
    test_noise = np.load(os.path.join(settings['dataset']['path'], 'test_noises.npy'))

    # get predictors and evaluation directories
    predictors = settings['predictors'].keys()

    # load evaluation results for each task
    results = {}
    for predictor in predictors:
        results[predictor] = {}
        for task in args.tasks:
            if task not in TASK_TO_PATH.keys():
                raise ValueError(f"Task {task} not supported")
            
            # if task in ['median_spectra', 'bounds_spectra'] and predictor == 'NS':
            #     continue
            results[predictor][task] = load_evaluation_results(settings['predictors'][predictor]['eval_dir'], task)


    # create dataframe with result
    # dataframes
    # rows: predictors
    # columns: metrics, scores, losses, etc.
    # print(results.keys())

    df_tasks = ['adc', 'losses', 'coverage', 'regression']
    pred = list(predictors)[0]

    # for each task, collect results of all predictors and create a specific csv
    for task in df_tasks:
        columns_labels = ['predictors']
        # columns_labels.extend(list(results[pred][task][0].keys()))

        res_ = results[pred][task][0]
        if task  in ['adc', 'losses']:
            columns_labels.extend(list(results[pred][task][0].keys()))
        elif task == 'coverage':
            for cl in list(res_.columns):
                columns_labels.append(f"avg_{cl}")
        elif task == 'regression':

            for k, v in res_.items():
                if isinstance(v, dict):
                    for kk, vv in v.items():
                        columns_labels.append(f"{k}_{kk}")
                else:
                    columns_labels.append(k)
        else:
            raise ValueError(f"Task {task} not supported")

        records = []
        for predictor in predictors:
            res = results[predictor][task][0]
            task_result = [predictor]
            # print(task, predictor, type(res), res)

            if isinstance(res, dict):
                for k, v in res.items():

                    if isinstance(v, dict):
                        for kk, vv in v.items():
                            task_result.append(np.mean(vv))
                    elif isinstance(v, list): # Pinball
                        task_result.append(np.mean(v))
                    else:
                        task_result.append(np.mean(v))
            elif isinstance(res, pd.core.frame.DataFrame):
                df = res.to_numpy()
                task_result.extend(df.mean(axis=0))
            
            
            records.append(task_result)
        # print(columns_labels)
        df = pd.DataFrame(records, columns=columns_labels)
        df.set_index('predictors', inplace=True)
        df.to_csv(os.path.join(args.output_dir, f"{task}.csv"))

    
    # corner plot with multiple posteriors for each sample
    n_samples = test_targets.shape[0]
    
    posteriors = np.concatenate([results[predictor]['posterior'] for predictor in predictors], axis=0)
    
    colors = [v['color'] for k,v in settings['predictors'].items()]
    labels = [r"$\mathrm{R_p}$", r"$\mathrm{T_p}$", r"$\log \mathrm{H_2O}$",r"$\log \mathrm{CO_2}$", r"$\log \mathrm{CO}$",r"$\log \mathrm{CH_4}$", r"$\log \mathrm{NH_3}$"]
    # f0c571: gold
    # 1a80bb: blue
    print(colors)
    print(list(predictors))
    # corner plot with multiple full posteriors given a sample
    corner_plot_multiple_distributions(
        posteriors=posteriors,
        thetas=test_targets,
        labels=labels,
        colors=colors,
        model_labels=list(predictors),
        output_dir=os.path.join(args.output_dir, 'plots', 'samples')
    )

    # corner plot with multiple full posteriors 
    corner_plot_prior_posteriors(
        posteriors=posteriors,
        thetas=test_targets,
        labels=labels,
        colors=colors,
        model_labels=list(predictors),
        output_fname=os.path.join(args.output_dir, 'plots', 'corner_plot.pdf')
    )

    # diagrams with multiple posteriors
    plot_regression_diagrams(
        posteriors=posteriors,
        thetas=test_targets,
        labels=labels,
        colors=colors, # TODO: check correctness
        model_labels=list(predictors),
        output_dir=os.path.join(args.output_dir, 'plots', 'diagrams')
    )
    
    # plot multiple median spectra and bounds
    median_spectra = np.concatenate([results[predictor]['median_spectra'] for predictor in predictors], axis=0)
    bounds_spectra = np.concatenate([results[predictor]['bounds_spectra'] for predictor in predictors], axis=0)
    
    ariel_wlgrid, ariel_wlwidth, ariel_wngrid, ariel_wnwidth = ariel_resolution()
    plot_median_spectra_with_confidence_intervals(
        wl_grid=ariel_wlgrid,
        posterior_median_spectra=median_spectra,
        posterior_bounds_spectra=bounds_spectra,
        real_spectra=test_real_spectra,
        ideal_spectra=test_ideal_spectra,
        noises=test_noise,
        colors=colors,
        model_labels=list(predictors),
        output_dir=os.path.join(args.output_dir, 'plots', 'samples')
    )


    # save results
    with open(os.path.join(args.output_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)


    



