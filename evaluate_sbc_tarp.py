import argparse
import os
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import yaml
from utils.utils import load_fn
from metrics.sbc import run_sbc
from metrics.tarp import run_tarp
from sbi.analysis.plot import sbc_rank_plot, plot_tarp

from typing import Optional, List
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
mpl.rc('font',family='Times New Roman')

TASK_TO_PATH = {
    "sbc": os.path.join("posterior_distribution.npy"),
    "tarp": os.path.join("posterior_distribution.npy"),
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


def perform_sbc_plot(
    posterior_samples: torch.Tensor,
    thetas: torch.Tensor,
    plot_type: str = "cdf",
    labels: List[str] = None,
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None,
):

    num_samples, num_posterior_samples, dim_theta = posterior_samples.shape
    
    # run the SBC evaluation
    ranks, _ = run_sbc(
        thetas,
        posterior_samples,
    )

    f, ax = sbc_rank_plot(
        ranks=ranks,
        num_posterior_samples=num_posterior_samples,
        plot_type=plot_type,
        parameter_labels=labels,
        num_bins=None,  # by passing None we use a heuristic for the number of bins.
        fig=fig,
        ax=ax
    )
    return f, ax

def plot_multiple_tarp(
    ecp: np.ndarray,
    alpha: np.ndarray,
    color: str = "blue",
    label: Optional[str] = None,
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None,
):
    if fig is None and ax is None: 
        fig = plt.figure(figsize=(6, 6))
        ax: Axes = plt.gca()
        ax.plot(alpha, alpha, color="black", linestyle="--", label="ideal")
        

    ax.plot(alpha, ecp, color=color, label=label)
    ax.set_xlabel(r"Credibility Level $\gamma$")
    ax.set_ylabel(r"Expected Coverage Probability")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    # ax.set_title("TARP")    
    return fig, ax  # type: ignore


def perform_tarp_plot(
    posterior_samples: np.ndarray,
    thetas: np.ndarray,
    model_label: str,
    color: str = "blue",
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None,
):
    # tarp expects (n_posterior_samples, n_tarp_samples, n_dims) for posterior_samples
    
    posterior_samples = posterior_samples.permute(1, 0, 2)
    # print(f"posterior_samples shape: {posterior_samples.shape}")
    
    ecp, alpha = run_tarp(
        thetas,
        posterior_samples,
        references=None,  # will be calculated automatically.
    )

    # Similar to SBC, we can check then check whether the distribution of ecp is close to
    # that of alpha.

    # Or, we can perform a visual check.
    fig, ax = plot_multiple_tarp(
        ecp, 
        alpha,
        color=color,
        label=model_label,
        fig=fig,
        ax=ax,
        )
    return fig, ax

if __name__ == "__main__":
    tasks = ['sbc', 'tarp']
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings_file", required=True, help="Settings file for the comparrison")
    ap.add_argument("--tasks", nargs='+', default=tasks, help="Tasks to compare")
    ap.add_argument("--output_dir", required=True, help="Output directory for the comparison")
    args = ap.parse_args()


    # create output directory
    if not os.path.exists(args.output_dir):
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # load settings file for comparison
    with open(args.settings_file, 'r') as f:
        settings = yaml.safe_load(f)

    # load test dataset
    test_targets = np.load(os.path.join(settings['dataset']['path'], 'test_targets.npy'))
    n_samples, n_targets = test_targets.shape

    TASK_TO_DATA = {
        'posterior': test_targets,
    }

    # get predictors and evaluation directories
    predictors = list(settings['predictors'].keys())
    print(f"Predictors: {predictors}")
    colors = [v['color'] for k,v in settings['predictors'].items()]
    labels = [r"$\mathrm{R_p}$", r"$\mathrm{T_p}$", r"$\log \mathrm{H_2O}$",r"$\log \mathrm{CO_2}$", r"$\log \mathrm{CO}$",r"$\log \mathrm{CH_4}$", r"$\log \mathrm{NH_3}$"]


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


    # sbc cdf plot
    fig, axs = plt.subplots(1, len(predictors), figsize=(18, 9))
    perform_sbc_plot(
        torch.from_numpy(results[predictors[0]]['sbc'][0]), 
        torch.from_numpy(test_targets), 
        labels=labels,
        fig=fig,
        ax=axs[0],
    )
    axs[0].legend()
    axs[0].set_title(f"{predictors[0]}")
    
    for predictor, color, ax in zip(predictors[1:], colors[1:], axs[1:]):
        
        perform_sbc_plot(
            torch.from_numpy(results[predictor]['sbc'][0]), 
            torch.from_numpy(test_targets), 
            labels=labels,
            fig=fig,
            ax=ax,
        )
        ax.legend()
        ax.set_title(f"{predictor}")
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, 'sbc_cdf.png'), dpi=400)   
    fig.savefig(os.path.join(args.output_dir, 'sbc_cdf.pdf'), format='pdf', bbox_inches='tight', dpi=400)
    plt.close(fig)
 

    # sbc rank plot
    fig, axs = plt.subplots(len(predictors), n_targets, figsize=(18, 24))
    perform_sbc_plot(
        torch.from_numpy(results[predictors[0]]['sbc'][0]), 
        torch.from_numpy(test_targets), 
        plot_type='hist',
        labels=labels,
        fig=fig,
        ax=axs[0],
    )

    axs[0, 0].set_ylabel(f"{predictors[0]}")

    for idx, ax in enumerate(axs[0]):
        ax.legend([f"rank set {idx}"])
    
    for predictor, color, ax in zip(predictors[1:], colors[1:], axs[1:]):
        
        perform_sbc_plot(
            torch.from_numpy(results[predictor]['sbc'][0]), 
            torch.from_numpy(test_targets), 
            plot_type='hist',
            labels=labels,
            fig=fig,
            ax=ax,
        )
        ax[0].set_ylabel(f"{predictor}")
        for idx, axx in enumerate(ax):
            axx.legend([f"rank set {idx}"])
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, 'sbc_hist.png'), dpi=400) 
    fig.savefig(os.path.join(args.output_dir, 'sbc_hist.pdf'), format='pdf', bbox_inches='tight', dpi=400)
    plt.close(fig)
    
        
    # tarp plot
    fig, ax = perform_tarp_plot(
        torch.from_numpy(results[predictors[0]]['tarp'][0]), 
        torch.from_numpy(test_targets), 
        predictors[0],
        color=colors[0],
    )
    for predictor, color in zip(predictors[1:], colors[1:]): 
        fig, ax = perform_tarp_plot(
            torch.from_numpy(results[predictor]['tarp'][0]), 
            torch.from_numpy(test_targets), 
            predictor,
            color=color,
            fig=fig,
            ax=ax,
        )
    ax.legend()
    ax.grid()
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, 'tarp.png'), dpi=400)
    fig.savefig(os.path.join(args.output_dir, 'tarp.pdf'), format='pdf', bbox_inches='tight', dpi=400)
    plt.close(fig)
            
            
