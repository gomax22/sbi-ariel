import argparse
import os
from tkinter import font
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import yaml
from sklearn.metrics import median_absolute_error, mean_squared_error, mean_absolute_error, root_mean_squared_error
from utils.utils import load_fn
import pprint
import matplotlib as mpl
mpl.rc('font',family='Times New Roman')
mpl.rcParams['text.usetex'] = False

TASK_TO_PATH = {
    'posterior': os.path.join("posterior_distribution.npy"),
    'real_spectra': os.path.join("evaluation", "adc", "median_spectra", "posterior_median_spectra.npy"),
    'ideal_spectra': os.path.join("evaluation", "adc", "median_spectra", "posterior_median_spectra.npy"),
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
    tasks = ['posterior', 'real_spectra', 'ideal_spectra']
    task_labels = [r"Target parameters ($\theta_{\mathrm{in}}$)", r"Real Spectra ($\tilde{x}$)", r"Ideal Spectra ($x$)"]
    # metrics = ['mse', 'mae', 'med_ae', 'rmse', 'mape', 'r2']
    metrics = ['MSE', 'MAE', 'MedAE', 'RMSE']
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings_file", required=True, help="Settings file for the comparrison")
    ap.add_argument("--tasks", nargs='+', default=tasks, help="Tasks to compare")
    ap.add_argument("--output_dir", default='comparison/losses', help="Output directory for the comparison")
    args = ap.parse_args()


    # create output directory
    if not os.path.exists(args.output_dir):
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # load settings file for comparison
    with open(args.settings_file, 'r') as f:
        settings = yaml.safe_load(f)

    # load test dataset
    test_targets = np.load(os.path.join(settings['dataset']['path'], 'test_targets.npy'))[:, None, :]
    n_samples = test_targets.shape[0]

    test_targets = np.tile(test_targets, (1, settings['evaluation']['n_repeats'], 1))
    test_ideal_spectra = np.load(os.path.join(settings['dataset']['path'], 'test_ideal_spectra.npy'))
    test_real_spectra = np.load(os.path.join(settings['dataset']['path'], 'test_real_spectra.npy'))
    
    TASK_TO_DATA = {
        'ideal_spectra': test_ideal_spectra,
        'real_spectra': test_real_spectra,
        'posterior': test_targets,
    }
    TASK_TO_LABEL = {k:v for k,v in zip(args.tasks, task_labels)}
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


    # create dataframe storing the mse, mae, rmse, mape and r2 scores
    # for each predictor

    losses = {}
    distributions = {}
    pbar = tqdm(total=len(predictors) * len(args.tasks) * len(metrics), desc="Computing losses...")
    for predictor in predictors:
        losses[predictor], distributions[predictor] = {}, {}
        for task in args.tasks:
            losses[predictor][task], distributions[predictor][task] = {}, {}
            for metric in metrics:

                # compute mse
                if metric == 'MSE':
                    distributions[predictor][task][metric] = mean_squared_error(
                        TASK_TO_DATA[task].reshape(n_samples, -1), 
                        results[predictor][task][0].reshape(n_samples, -1),
                        multioutput='raw_values',
                    )
                    
                elif metric == 'MAE':
                    distributions[predictor][task][metric] = mean_absolute_error(
                        TASK_TO_DATA[task].reshape(n_samples, -1), 
                        results[predictor][task][0].reshape(n_samples, -1),
                        multioutput='raw_values',
                    )
                    
                elif metric == 'RMSE':
                    distributions[predictor][task][metric] = root_mean_squared_error(
                        TASK_TO_DATA[task].reshape(n_samples, -1),
                        results[predictor][task][0].reshape(n_samples, -1),
                        multioutput='raw_values',
                    )
                elif metric == 'MedAE':
                    distributions[predictor][task][metric] = median_absolute_error(
                        TASK_TO_DATA[task].reshape(n_samples, -1),
                        results[predictor][task][0].reshape(n_samples, -1),
                        multioutput='raw_values',
                    )
                else:
                    raise ValueError(f"Metric {metric} not supported")
                losses[predictor][task][metric] = np.mean(distributions[predictor][task][metric])
                
                pbar.update(1)
    pbar.close()
    # create dataframe with result
    # columns = metrics
    # rows = predictors
    colors = [v['color'] for k,v in settings['predictors'].items()]
    

    columns_labels = ['predictors'] + metrics
    for task in args.tasks:
        df = pd.DataFrame(columns=columns_labels)
        for predictor in predictors:
            row = [predictor]
            for metric in metrics:
                row.append(losses[predictor][task][metric])
            df.loc[len(df)] = row
        df.set_index('predictors', inplace=True)
        df.to_csv(os.path.join(args.output_dir, f"{task}_comparison.csv"))
    
    # plot the distributions of the losses as violins
    # for each predictor and task
    # put metrics on the y-axis and predictors on the x-axis
    
    error_distributions_dir = os.path.join(args.output_dir, "error_distributions")
    if not Path(error_distributions_dir).exists():
        Path(error_distributions_dir).mkdir(parents=True, exist_ok=True)
    
    for task in args.tasks:
        fig, axs = plt.subplots(len(metrics), 1, figsize=(len(predictors)* 2, len(metrics) * 2))
        fig.suptitle(f"Error Distributions for {TASK_TO_LABEL[task]}", fontsize=14)
        for i, metric in enumerate(metrics):
            ax = axs[i]
            data = []
            for predictor in predictors:
                data.append(distributions[predictor][task][metric])
            plots = ax.violinplot(data, showmeans=True, showmedians=True, showextrema=True)
            
            # Make all the violin statistics marks red:
            for pc, color in zip(plots['bodies'], colors):
                pc.set_facecolor(color)
                pc.set_edgecolor(color)
            # Set the color of the median lines
            for partname in ['cbars','cmins','cmaxes','cmeans','cmedians']:
                plots[partname].set_colors(colors)
            
            ax.set_ylabel(metric, fontsize=14)
            ax.set_xticks(range(1, len(predictors) + 1))
            ax.set_xticklabels(predictors, fontsize=14)
            ax.set_yscale('log')
            ax.grid()
        fig.tight_layout()
        fig.savefig(os.path.join(error_distributions_dir, f"{task}_error_distributions.png"), dpi=400)
        fig.savefig(os.path.join(error_distributions_dir, f"{task}_error_distributions.pdf"), format='pdf', bbox_inches='tight', dpi=400)
        