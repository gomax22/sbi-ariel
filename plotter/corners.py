import numpy as np
import corner
import matplotlib.pyplot as plt
from typing import List
from sympy import plot
from tqdm import tqdm
from pathlib import Path
import os
import sys
sys.path.append("../")
from metrics.coverage import compute_confidence_interval
import random
import torch
from matplotlib.lines import Line2D
import matplotlib as mpl
mpl.rcParams['text.usetex'] = False 
mpl.rc('font',family='Times New Roman')


SEED = 123
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False




def corner_plot_multiple_distributions(
    posteriors: np.ndarray,
    thetas: np.ndarray,
    labels: List[str],
    colors: List[str],
    model_labels: List[str],
    output_dir: str
    ):  

    num_samples, num_targets = thetas.shape

    if not Path(output_dir).exists():
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    for idx in tqdm(range(num_samples), desc="Plotting corners..."):
        theta = thetas[idx]
        posterior_samples = posteriors[:, idx]


        sample_output_dir = os.path.join(output_dir, f"sample_{idx}")
        if not Path(sample_output_dir).exists():
            Path(sample_output_dir).mkdir(parents=True, exist_ok=True)

        fig = corner.corner(
            posterior_samples[0], 
            labels=labels, 
            label_kwargs={"fontsize": 12, "fontname": "Times New Roman"},
            hist_kwargs={
                "histtype": "stepfilled", 
                "alpha": 0.2, 
                "color": colors[0]
            },
            hist2d_kwargs={
                "levels": (0.68, 0.95),
                "no_fill_contours": True,
                "plot_datapoints": False,
                "plot_density": False,
                "plot_contours": False,
                "pcolor_kwargs": {"alpha": 0.2},
                "contour_kwargs": {"alpha": 0.2, "lw": 0.5},
                "contourf_kwargs": {"alpha": 0.2, "lw": 0.5},
                "alpha": 0.2, },
            bins=50,
            show_titles=True, 
            color=colors[0], 
            title_kwargs={"fontsize": 12, "fontname": "Times New Roman"},
        )
        # fig = corner.corner(posteriors[0], labels=labels, show_titles=False, title_kwargs={"fontsize": 10})
        # Extract the axes
        axes = np.array(fig.axes).reshape((num_targets, num_targets))
        
        for posterior, color in zip(posterior_samples[1:], colors[1:]):
            corner.corner(
                posterior, 
                labels=labels, 
                hist_kwargs={
                    "histtype": "stepfilled", 
                    "alpha": 0.2, 
                    "color": color,
                },
                hist2d_kwargs={
                    "levels": (0.68, 0.95),
                    "no_fill_contours": True,
                    "plot_datapoints": False,
                    "plot_density": False,
                    "plot_contours": False,
                    "pcolor_kwargs": {"alpha": 0.2},
                    "contour_kwargs": {"alpha": 0.2, "lw": 0.5},
                    "contourf_kwargs": {"alpha": 0.2, "lw": 0.5},
                    "alpha": 0.2, },
                color=color, 
                bins=50,
                levels=(0.68, 0.95),
                show_titles=False, 
                fig=fig
            )

            # compute confidence intervals
            # ci_one_sigma = compute_confidence_interval(posterior, 0.68)
            # ci_two_sigma = compute_confidence_interval(posterior, 0.95)

            # # Loop over the diagonal
            # for i in range(n_targets):
            #     ax = axes[i, i]
            #     # plot 1-sigma confidence intervals for model
            #     ax.axvline(ci_one_sigma[0][i], color=color, linestyle="--")
            #     ax.axvline(ci_one_sigma[1][i], color=color, linestyle="--")

            #     # plot 2-sigma confidence intervals for model
            #     ax.axvline(ci_two_sigma[0][i], color=color, linestyle=":")
            #     ax.axvline(ci_two_sigma[1][i], color=color, linestyle=":")
        
        # Loop over the diagonal
        for i in range(num_targets):
            ax = axes[i, i]
            ax.axvline(theta[i].item(), linestyle="--", color="g")
            x_lim = ax.get_xlim()
            ax.set_xlim(min(x_lim[0], theta[i].item()) - 0.2, max(x_lim[1], theta[i].item()) + 0.2) # this offset makes sense in the standardized domain (e.g. for T_p it's useless)

        # Loop over the histograms
        for yi in range(num_targets): # yi 
            for xi in range(yi):
                ax = axes[yi, xi]
                # ax.set_title(f"yi: {yi}, xi: {xi}")
                ax.axvline(theta[xi].item(), linestyle="--", color="g")
                
                # ax.axvline(value2[xi], color="r")
                ax.axhline(theta[yi].item(), linestyle="--", color="g")

                # ax.axhline(value2[yi], color="r")
                ax.plot(theta[xi].item(), theta[yi].item(), "sg", markersize=4)
                # ax.plot(value2[xi], value2[yi], "sr")
                x_lim, y_lim = ax.get_xlim(), ax.get_ylim()
                ax.set_xlim(min(x_lim[0], theta[xi].item()) - 0.2, max(x_lim[1], theta[xi].item()) + 0.2) # this offset makes sense in the standardized domain
                ax.set_ylim(min(y_lim[0], theta[yi].item()) - 0.2, max(y_lim[1], theta[yi].item()) + 0.2)



        # plot legend in upper right corner of the figure
        handles = [Line2D([0], [0], lw=1.0, ls="--", color="green")]
        handles += [
            Line2D([0], [0], color=c, lw=4)
            for c in colors
        ]
        fig.legend(
            handles=handles,
            labels=[rf'Ground truth ($\theta^{{\mathrm{{in}}}}_{{{idx}}}$)'] + [label for label in model_labels],
            ncols=1,
            frameon=False,
            loc="upper right",
            fontsize=20,
            bbox_to_anchor=(0.95, 0.95),
        )
        
        # plt.figlegend(model_labels, fontsize='large')
        fig.savefig(os.path.join(sample_output_dir, f"corner_plot_{idx}.png"), dpi=400)
        fig.savefig(os.path.join(sample_output_dir, f"corner_plot_{idx}.pdf"), format='pdf', bbox_inches='tight', dpi=400)
        plt.close(fig)


        # if idx == 0:
        #     break

    

def corner_plot_single_distribution(
    posterior_samples: np.ndarray,
    theta: np.ndarray,
    labels: List,
    model_label: str,
    output_fname: str,
    ):

    # compute confidence intervals
    ci_one_sigma = compute_confidence_interval(posterior_samples, 0.68)
    ci_two_sigma = compute_confidence_interval(posterior_samples, 0.95)

    n_targets = posterior_samples.shape[-1]

    try:
        fig = corner.corner(
            posterior_samples, 
            labels=labels, 
            label_kwargs={"fontsize": 12, "fontname": "Times New Roman"},
            show_titles=True, 
            color="blue",
            title_kwargs={"fontsize": 12, "fontname": "Times New Roman"},)

        # Extract the axes
        axes = np.array(fig.axes).reshape((n_targets, n_targets))
        
        # Loop over the diagonal
        for i in range(n_targets):
            ax = axes[i, i]
            ax.axvline(theta[i].item(), color="g")
            
            # plot 1-sigma confidence intervals for model
            ax.axvline(ci_one_sigma[0][i], color="b", linestyle="--")
            ax.axvline(ci_one_sigma[1][i], color="b", linestyle="--")

            # plot 2-sigma confidence intervals for model
            ax.axvline(ci_two_sigma[0][i], color="b", linestyle=":")
            ax.axvline(ci_two_sigma[1][i], color="b", linestyle=":")

            x_lim = ax.get_xlim()
            ax.set_xlim(min(x_lim[0], theta[i].item()) - 0.2, max(x_lim[1], theta[i].item()) + 0.2) # this offset makes sense in the standardized domain (e.g. for T_p it's useless)
        
        # Loop over the histograms
        for yi in range(n_targets): # yi 
            for xi in range(yi):
                ax = axes[yi, xi]
                # ax.set_title(f"yi: {yi}, xi: {xi}")
                ax.axvline(theta[xi].item(), color="g")
                
                # ax.axvline(value2[xi], color="r")
                ax.axhline(theta[yi].item(), color="g")

                # ax.axhline(value2[yi], color="r")
                ax.plot(theta[xi].item(), theta[yi].item(), "sg")
                # ax.plot(value2[xi], value2[yi], "sr")
                x_lim, y_lim = ax.get_xlim(), ax.get_ylim()
                ax.set_xlim(min(x_lim[0], theta[xi].item()) - 0.2, max(x_lim[1], theta[xi].item()) + 0.2) # this offset makes sense in the standardized domain
                ax.set_ylim(min(y_lim[0], theta[yi].item()) - 0.2, max(y_lim[1], theta[yi].item()) + 0.2)
        
        # plot legend in upper right corner of the figure
        plt.figlegend([model_label, r"$\theta_{\mathrm{in}}$"], fontsize='large')
        
        # both with theta
        fig.savefig(output_fname, dpi=400)
        plt.close(fig)
    except Exception as e:
        print(e)
        print("Error in corner plot")



def corner_plot(
    posterior_samples: np.ndarray, 
    reference_samples: np.ndarray, 
    theta: np.ndarray,
    model_ci_one_sigma: np.ndarray,
    model_ci_two_sigma: np.ndarray,
    ns_ci_one_sigma: np.ndarray,
    ns_ci_two_sigma: np.ndarray,
    labels: List,
    model_label: str,
    output_fname: str,
    ):


    n_targets = reference_samples.shape[-1]
    fig = corner.corner(
        posterior_samples, 
        labels=labels, 
        label_kwargs={"fontsize": 12, "fontname": "Times New Roman"},
        show_titles=True, 
        color="blue", 
        title_kwargs={"fontsize": 12, "fontname": "Times New Roman"},
    )
    corner.corner(
        reference_samples, 
        labels=labels, 
        label_kwargs={"fontsize": 12, "fontname": "Times New Roman"},
        color="red", 
        show_titles=False, 
        fig=fig
    )
    
    # Extract the axes
    axes = np.array(fig.axes).reshape((n_targets, n_targets))
    
    # Loop over the diagonal
    for i in range(n_targets):
        ax = axes[i, i]
        ax.axvline(theta[i].item(), color="g")
        
        # plot 1-sigma confidence intervals for model
        ax.axvline(model_ci_one_sigma[0][i], color="b", linestyle="--")
        ax.axvline(model_ci_one_sigma[1][i], color="b", linestyle="--")

        # plot 2-sigma confidence intervals for model
        ax.axvline(model_ci_two_sigma[0][i], color="b", linestyle=":")
        ax.axvline(model_ci_two_sigma[1][i], color="b", linestyle=":")

        # plot 1-sigma confidence intervals for NS
        ax.axvline(ns_ci_one_sigma[0][i], color="r", linestyle="--")
        ax.axvline(ns_ci_one_sigma[1][i], color="r", linestyle="--")
        
        # plot 2-sigma confidence intervals for NS
        ax.axvline(ns_ci_two_sigma[0][i], color="r", linestyle=":")
        ax.axvline(ns_ci_two_sigma[1][i], color="r", linestyle=":")
        
        x_lim = ax.get_xlim()
        ax.set_xlim(min(x_lim[0], theta[i].item()) - 0.2, max(x_lim[1], theta[i].item()) + 0.2) # this offset makes sense in the standardized domain (e.g. for T_p it's useless)
        # mean NS
        # mean model
        # ax.axvline(trace1[:, i].mean(), color="cyan")
        # ax.axvline(reference_sample[:, i].mean(), color="darkorange")
    
    # Loop over the histograms
    for yi in range(n_targets): # yi 
        for xi in range(yi):
            ax = axes[yi, xi]
            # ax.set_title(f"yi: {yi}, xi: {xi}")
            ax.axvline(theta[xi].item(), color="g")
            
            # ax.axvline(value2[xi], color="r")
            ax.axhline(theta[yi].item(), color="g")

            # ax.axhline(value2[yi], color="r")
            ax.plot(theta[xi].item(), theta[0, yi].item(), "sg")
            # ax.plot(value2[xi], value2[yi], "sr")
            x_lim, y_lim = ax.get_xlim(), ax.get_ylim()
            ax.set_xlim(min(x_lim[0], theta[xi].item()) - 0.2, max(x_lim[1], theta[xi].item()) + 0.2) # this offset makes sense in the standardized domain
            ax.set_ylim(min(y_lim[0], theta[yi].item()) - 0.2, max(y_lim[1], theta[yi].item()) + 0.2)
    
    # plot legend in upper right corner of the figure
    plt.figlegend([model_label, "NS", r"$\theta_{\mathrm{in}}$"], fontsize='large')
    
    # both with theta
    fig.savefig(output_fname, dpi=400)
    plt.close(fig)


def plot_distributions(
    posterior_samples: np.ndarray,
    reference_samples: np.ndarray,
    labels: List,
    model_label: str,
    output_fname: str,
    ):

    fig = corner.corner(posterior_samples, labels=labels, show_titles=True, color="blue", title_kwargs={"fontsize": 10})
    corner.corner(reference_samples, labels=labels, color="red", show_titles=False, fig=fig)

    # plot legend in upper right corner of the figure
    plt.figlegend([model_label, "NS"], fontsize='large')
    fig.savefig(output_fname, dpi=400)
    plt.close(fig)


def corner_plot_prior_posterior(
    posterior_samples: np.ndarray,
    thetas: np.ndarray,
    model_label: str,
    output_fname: str,
):
    # -------------------------------------------------
    # plot prior vs posteriors
    posterior_samples = posterior_samples.reshape(-1, posterior_samples.shape[-1])

    # take a random subset of size len(thetas)
    idx_model = np.random.choice(posterior_samples.shape[0], len(thetas), replace=False)
    posterior_samples = posterior_samples[idx_model]

    # make a corner plot of prior and posterior samples
    labels = [r"\mathrm{R_p}$", r"$\mathrm{T_p}$", r"$\log \mathrm{H_2O}$",r"$\log \mathrm{CO_2}$", r"$\log \mathrm{CO}$",r"$\log \mathrm{CH_4}$", r"$\log \mathrm{NH_3}$"]

    n_targets = thetas.shape[-1]
    fig = corner.corner(
        thetas, 
        labels=labels, 
        label_kwargs={"fontsize": 12, "fontname": "Times New Roman"},
        color="green", 
        show_titles=True, 
        title_kwargs={"fontsize": 10, "fontname": "Times New Roman"}
    )
    
    corner.corner(
        posterior_samples, 
        labels=labels, 
        show_titles=False, 
        color="blue", 
        fig=fig)   
    
    # Extract the axes
    axes = np.array(fig.axes).reshape((n_targets, n_targets))
        
    # Loop over the diagonal
    for i in range(n_targets):
        ax = axes[i, i]
        ax.axvline(thetas[:, i].mean().item(), color="g", linestyle="--")
        x_lim = ax.get_xlim()
        ax.set_xlim(min(x_lim[0], thetas[:, i].mean().item()) - 0.2, max(x_lim[1], thetas[:, i].mean().item()) + 0.2) # this offset makes sense in the standardized domain (e.g. for T_p it's useless)
        ax.axvline(posterior_samples[:, i].mean().item(), color="blue", linestyle="--")
    
    # Loop over the histograms
    for yi in range(n_targets): # yi 
        for xi in range(yi):
            ax = axes[yi, xi]
            # ax.set_title(f"yi: {yi}, xi: {xi}")
            ax.axvline(thetas[:, xi].mean().item(), color="g", linestyle="--")
            
            # ax.axvline(value2[xi], color="r")
            ax.axhline(thetas[:, yi].mean().item(), color="g", linestyle="--")
            
            # ax.axhline(value2[yi], color="r")
            ax.plot(thetas[:, xi].mean().item(), thetas[:, yi].mean().item(), "sg")
            # ax.plot(value2[xi], value2[yi], "sr")
            x_lim, y_lim = ax.get_xlim(), ax.get_ylim()
            ax.set_xlim(min(x_lim[0], thetas[:, xi].mean().item()) - 0.2, max(x_lim[1], thetas[:, xi].mean().item()) + 0.2) # this offset makes sense in the standardized domain
            ax.set_ylim(min(y_lim[0], thetas[:, yi].mean().item()) - 0.2, max(y_lim[1], thetas[:, yi].mean().item()) + 0.2)
            
    # plot legend in upper right corner of the figure
    plt.figlegend(["PRIOR", model_label], fontsize='large')
    
    # both with theta
    fig.savefig(output_fname, dpi=400)
    plt.close(fig)


def corner_plot_prior_posteriors(
    posteriors: np.ndarray,
    thetas: np.ndarray,
    labels: List[str],
    colors: List[str],
    model_labels: List[str],
    output_fname: str
):
    # -------------------------------------------------
    # plot prior vs posteriors
    thetas = thetas.reshape(-1, thetas.shape[-1])
    posteriors = np.reshape(posteriors, (len(posteriors), -1, posteriors[0].shape[-1]))

    # take a random subset of size len(thetas)
    random_indices = np.random.choice(thetas.shape[0], len(thetas), replace=False)
    thetas = thetas[random_indices]
    posteriors = posteriors[:, random_indices]

    # make a corner plot of prior and posterior samples
    n_targets = thetas.shape[-1]
    fig = corner.corner(
        thetas,
        labels=labels,
        label_kwargs={"fontsize": 12, "fontname": "Times New Roman"},
        color="green",
        show_titles=True, 
        title_kwargs={"fontsize": 12, "fontname": "Times New Roman"}
    )
    
    for posterior, color in zip(posteriors, colors):
        corner.corner(
            posterior, 
            labels=labels, 
            label_kwargs={"fontsize": 12, "fontname": "Times New Roman"},
            color=color, 
            show_titles=False, 
            fig=fig
        )   
    
    # Extract the axes
    axes = np.array(fig.axes).reshape((n_targets, n_targets))
        
    # Loop over the diagonal
    for i in range(n_targets):
        ax = axes[i, i]
        ax.axvline(thetas[:, i].mean().item(), color="g", linestyle="--")
        x_lim = ax.get_xlim()
        ax.set_xlim(min(x_lim[0], thetas[:, i].mean().item()) - 0.2, max(x_lim[1], thetas[:, i].mean().item()) + 0.2) # this offset makes sense in the standardized domain (e.g. for T_p it's useless)
        for posterior, color in zip(posteriors, colors):
            ax.axvline(posterior[:, i].mean().item(), color=color, linestyle="--")
    
    # Loop over the histograms
    for yi in range(n_targets): # yi 
        for xi in range(yi):
            ax = axes[yi, xi]
            # ax.set_title(f"yi: {yi}, xi: {xi}")
            ax.axvline(thetas[:, xi].mean().item(), color="g", linestyle="--")
            
            # ax.axvline(value2[xi], color="r")
            ax.axhline(thetas[:, yi].mean().item(), color="g", linestyle="--")
            
            for posterior, color in zip(posteriors, colors):
                ax.axvline(posterior[:, xi].mean().item(), color=color, linestyle="--")
                ax.axhline(posterior[:, yi].mean().item(), color=color, linestyle="--")
            
            # ax.axhline(value2[yi], color="r")
            ax.plot(thetas[:, xi].mean().item(), thetas[:, yi].mean().item(), "sg")
            # ax.plot(value2[xi], value2[yi], "sr")
            x_lim, y_lim = ax.get_xlim(), ax.get_ylim()
            ax.set_xlim(min(x_lim[0], thetas[:, xi].mean().item()) - 0.2, max(x_lim[1], thetas[:, xi].mean().item()) + 0.2)
            ax.set_ylim(min(y_lim[0], thetas[:, yi].mean().item()) - 0.2, max(y_lim[1], thetas[:, yi].mean().item()) + 0.2)

    # plot legend in upper right corner of the figure
    handles = [Line2D([0], [0], lw=4, color="green")]
    handles += [
        Line2D([0], [0], color=c, lw=4)
        for c in colors
    ]
    fig.legend(
        handles=handles,
        labels=[r"$\theta_{\mathrm{in}}$"] + [label for label in model_labels],
        ncols=1,
        frameon=False,
        loc="upper right",
        fontsize=20,
        bbox_to_anchor=(0.95, 0.95),
    )
    # plt.figlegend(model_labels, fontsize='large')
    fig.savefig(output_fname, dpi=400)
    fig.savefig(output_fname, format='pdf', bbox_inches='tight', dpi=400)
    
    plt.close(fig)
