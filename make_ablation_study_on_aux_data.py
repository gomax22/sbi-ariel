import os
import numpy as np
import argparse
from utils.utils import load_fn
from pprint import pprint
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import matplotlib as mpl
mpl.rcParams['text.usetex'] = False 
mpl.rc('font',family='Times New Roman')


TASK_TO_PATH = {
    "adc": "adc.csv",
    "losses": "losses.csv",
    "regression": "regression.csv",
    "coverage": "coverage.csv",
    "std": "coverage.csv",
    "posterior": "posterior_comparison.csv",
    "real_spectra": "real_spectra_comparison.csv",
    "ideal_spectra": "ideal_spectra_comparison.csv",
}


def load_evaluation_results(
    eval_dir: str,
    task: str,
    **kwargs
):
    eval_path = TASK_TO_PATH[task]
    fpath = os.path.join(eval_dir, eval_path)
        
    ext = fpath.split('.')[-1]
    load_func = load_fn(ext)
    
    return load_func(fpath, **kwargs)

if __name__ == "__main__":
    tasks = ['adc', 'posterior', 'real_spectra', 'std', 'regression', 'coverage']
    without_aux = ['IUN-P', 'IUN-A', 'IUS-P', 'IUS-A', 'RUN-P', 'RUN-A', 'RUS-P', 'RUS-A']
    with_aux = ['IUAN-P', 'IUAN-A', 'IUAS-P', 'IUAS-A', 'RUAN-P', 'RUAN-A', 'RUAS-P', 'RUAS-A']
    filter_columns = {
        'adc': {
            'avg_posterior_score': "Posterior\nScore",
            'avg_spectral_score': "Spectral\nScore",
            'final_score': "Final\nScore",
        },
        'regression': {
            'NLL_independent': "NLL", 
            'QCE_independent': "QCE", 
            'Pinball': r"$\mathcal{L}_{\mathrm{Pin}}$", 
            'ENCE_independent': "ENCE", 
            'UCE_independent': "UCE", 
            'NLL_joint': "NLL\n(joint)", 
            'QCE_joint': "QCE\n(joint)",
        },
        'losses': {
            'avg_posterior_theta_loss': r"$\theta^{\mathrm{in}}$", 
            'avg_posterior_ideal_spectrum_loss': r"$x$", 
            'avg_posterior_real_spectrum_loss': r"$\tilde{x}$",
        },
        'coverage': {
            'avg_posterior_accuracy_ratio_one_sigma': "MCR\n(1s)", 
            'avg_posterior_accuracy_ratio_two_sigma': "MCR\n(2s)",
            'avg_posterior_accuracy_ratio_support': "MCR\n(supp.)",
            'avg_posterior_accuracy_all_one_sigma': "JCR\n(1s)",
            'avg_posterior_accuracy_all_two_sigma': "JCR\n(2s)",
            'avg_posterior_accuracy_all_support': "JCR\n(supp.)",
        },
        'std': {
            'avg_posterior_std_R_p': r"$\mathrm{R_p}$",
            'avg_posterior_std_T_p': r"$\mathrm{T_p}$", 
            'avg_posterior_std_log_H2O': r"$\log \mathrm{H_2O}$",
            'avg_posterior_std_log_CO2': r"$\log \mathrm{CO_2}$",
            'avg_posterior_std_log_CO': r"$\log \mathrm{CO}$",
            'avg_posterior_std_log_CH4': r"$\log \mathrm{CH_4}$",
            'avg_posterior_std_log_NH3': r"$\log \mathrm{NH_3}$",
        },
        'posterior': {
            'MSE': "MSE",
            'MAE': "MAE",
            'MedAE': "MedAE",
            'RMSE': "RMSE",
        },
        'real_spectra': {
            'MSE': "MSE",
            'MAE': "MAE",
            'MedAE': "MedAE",
            'RMSE': "RMSE",
        },
        'ideal_spectra': {
            'MSE': "MSE",
            'MAE': "MAE",
            'MedAE': "MedAE",
            'RMSE': "RMSE",
        },
    }
    titles = {
        'adc': 'Scores of the Ariel Data Challenge',
        'losses': 'Prediction errors',
        'std': 'Posterior Standard Deviations',
        'regression': 'Calibration',
        'coverage': 'Posterior Coverage',
        'posterior': r'Prediction errors on atmospheric parameters ($\theta^{\mathrm{in}}$)',
        'ideal_spectra': r'Prediction errors on ideal spectra ($x$)',
        'real_spectra': r'Prediction errors on real spectra ($\tilde{x}$)',
    }

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Ablation study on auxiliary data")
    parser.add_argument(
        "--comparison_dir",
        type=str,
        required=True, 
        help="Path to the directory containing the comparison results"
    )
    parser.add_argument(
        "--tasks",
        nargs='+',
        default=tasks,
        help="Tasks to compare"
    )
    args = parser.parse_args()
    tasks = args.tasks

    # Load the comparison results
    results = {}
    for task in tasks:
        results[task] = load_evaluation_results(args.comparison_dir, task, index_col=0)

    performance = {
        "without_aux": {},
        "with_aux": {}
    }
    
    for task in tasks:
        keys = list(filter(lambda x: x in filter_columns[task].keys(), list(results[task].keys())))
        print(f"Task: {task}, Keys: {keys}")
        performance["without_aux"][task] = {k:v for k,v in zip(keys, results[task].loc[without_aux].filter(items=keys).to_numpy().mean(axis=0))}
        performance["with_aux"][task] = {k:v for k,v in zip(keys, results[task].loc[with_aux].filter(items=keys).to_numpy().mean(axis=0))}

    pprint(performance)

    # compute performance gain (in %) of using auxiliary data
    performance_gain = {}

    for task in tasks:
        performance_gain[task] = {}
        for key in performance["without_aux"][task].keys():
            performance_gain[task][key] = (performance["with_aux"][task][key] - performance["without_aux"][task][key]) / performance["without_aux"][task][key] * 100
        #performance_gain[task] = {k:v for k,v in sorted(performance_gain[task].items(), key=lambda item: item[1], reverse=True)}
    
    print("\n\nRelative performance (in %):")
    pprint(performance_gain)

    # make a bar plot for each task
    # display in red and green the performance of the models without and with auxiliary data, respectively
    # fig, axs = plt.subplots(2, 2, figsize=(20, 12), sharey=True)
    # fig, axs = plt.subplots(1, 5, figsize=(28, 6), sharey=True)

    # for task in tasks:
    #     ax = axs.flatten()[tasks.index(task)]
    #     species = [filter_columns[task][key] for key in list(performance_gain[task].keys())]
    #     values = list(performance_gain[task].values())



    #     # Create bar plot
    #     x = np.arange(len(species))
    #     multiplier = 0

    #     # Plot values in red if negative, green if positive
    #     if task in ['adc', 'coverage']:
    #         colors = ['red' if v < 0 else 'green' for v in values]
    #     else:
    #         colors = ['green' if v < 0 else 'red' for v in values]

    #     # Add grid lines
    #     ax.set_axisbelow(True)
    #     ax.yaxis.grid(color='gray', linestyle='dashed')
    #     ax.yaxis.set_tick_params(labelsize=18)

    #     # rects1 = ax.bar(species, values, width, label='w/o auxiliary data', color=colors)
    #     ax.bar(species, values, color=colors, alpha=0.7, edgecolor='black')
    #     ax.set_ylabel('Relative Performance (%)', fontsize=20)
    #     ax.set_title(titles[task], fontsize=20)

    #     ax.set_xticks(x, species, fontsize=20)
    #     # ax.set_ylim(-np.abs(values).max() - 10, np.abs(values).max() + 10)


    # # Add legend
    # handles = [
    #     Line2D([0], [0], color='red', lw=4.0),
    #     Line2D([0], [0], color='green', lw=4.0)
    # ]
    # fig.legend(
    #     handles=handles,
    #     labels=['Performance drop', 'Performance gain'],
    #     ncols=2,
    #     frameon=False,
    #     fontsize=20,
    #     loc="outside upper center",
    # )
    # fig.tight_layout()
    # fig.subplots_adjust(
    #     top=0.85,
    #     bottom=0.15
    # )


    fig, axs = plt.subplots(2, 3, figsize=(26, 16), sharey=True)

    for task in tasks:
        ax = axs.flatten()[tasks.index(task)]
        species = [filter_columns[task][key] for key in list(performance_gain[task].keys())]
        values = list(performance_gain[task].values())

        # Create bar plot
        x = np.arange(len(species))
        multiplier = 0

        # Plot values in red if negative, green if positive
        if task in ['adc', 'coverage']:
            colors = ['red' if v < 0 else 'green' for v in values]
        else:
            colors = ['green' if v < 0 else 'red' for v in values]

        # Add grid lines
        ax.set_axisbelow(True)
        ax.yaxis.grid(color='gray', linestyle='dashed')
        ax.yaxis.set_tick_params(labelsize=18)

        # rects1 = ax.bar(species, values, width, label='w/o auxiliary data', color=colors)
        ax.bar(species, values, color=colors, alpha=0.7, edgecolor='black')
        ax.set_ylabel('Relative Performance (%)', fontsize=24)
        ax.set_title(titles[task], fontsize=24)

        ax.set_xticks(x, species, fontsize=20)
        # ax.set_ylim(-np.abs(values).max() - 10, np.abs(values).max() + 10)


    # Add legend
    handles = [
        Line2D([0], [0], color='red', lw=4.0),
        Line2D([0], [0], color='green', lw=4.0)
    ]
    fig.legend(
        handles=handles,
        labels=['Performance drop', 'Performance gain'],
        ncols=2,
        frameon=False,
        fontsize=24,
        loc="outside upper center",
    )
    fig.tight_layout()
    fig.subplots_adjust(
        top=0.93,
        bottom=0.07
    )

    # Save the figure (path already exists)
    fig.savefig(os.path.join(args.comparison_dir, "performance_gain.pdf"), format='pdf', bbox_inches='tight', dpi=400)
    fig.savefig(os.path.join(args.comparison_dir, "performance_gain.png"), format='png', bbox_inches='tight', dpi=400)
    plt.close(fig)


