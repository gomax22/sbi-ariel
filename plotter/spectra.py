from typing import List
from matplotlib.lines import Line2D
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
import os
import matplotlib as mpl
from tqdm import tqdm
mpl.rcParams['text.usetex'] = False 

def plot_spectrum_with_confidence_interval(ax, wl_grid, spectrum, ci, alphas, color, label):
    ax.plot(wl_grid, spectrum, color=color, alpha=alphas[0], label=label, zorder=2)
    # ax.fill_between(ariel_wngrid, spectrum - noise, spectrum + noise, alpha=0.2, color="green", zorder=2) # 100% confidence interval
    ax.fill_between(wl_grid, spectrum - ci, spectrum + ci, color=color, alpha=alphas[1], zorder=2) # 50% confidence interval
    return ax


def plot_median_spectrum_with_confidence_intervals(
    wl_grid,
    posterior_spectrum: np.ndarray,
    posterior_bounds: np.ndarray,
    real_spectrum: np.ndarray,
    ideal_spectrum: np.ndarray,
    noise: np.ndarray,
    model_label: str,
    output_fname: str
    ):

    ci_spectrum = stats.norm.interval(0.5)
    ci_spectrum = np.abs(np.array(ci_spectrum[0]) * noise)

    # plot the median spectra and iqr ranges
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.grid(alpha=0.5)

    ax.plot(wl_grid, real_spectrum, color="black", alpha=0.5, label="real")
    ax = plot_spectrum_with_confidence_interval(
        ax, wl_grid, ideal_spectrum, ci_spectrum, alphas=[0.5, 0.2], color="green", label="ideal"
    )


    ax = plot_spectrum_with_confidence_interval(
        ax, wl_grid, posterior_spectrum, posterior_bounds, alphas=[0.5, 0.2], color="blue", label=model_label
    )
    
    ax.legend(fontsize='large')
    ax.set_ylabel(r"Transit depth [$\mathrm{R_p^2}/\mathrm{R_*^2}$]")
    
    ax.set_xlim(np.min(wl_grid) - 0.05 * np.min(wl_grid), np.max(wl_grid) + 0.05 * np.max(wl_grid))
    ax.set_xlabel(r"Wavelength [$\mathrm{\mu m}$]")

    if np.max(wl_grid) - np.min(wl_grid) > 5:
        ax.set_xscale("log")
        ax.tick_params(axis="x", which="minor")
    
    ## multiple by 100 to turn it into percentage. 
    # spectrum[idx, 0, :] => wl_grid
    # spectrum[idx, 1, :] => observed spectrum
    # spectrum[idx, 2, :] => noise
    # plt.errorbar(x=spectrum[:,0], y= spectrum[:,1]*100, yerr=spectrum[:,2]*100 )
    # fig.savefig(output_fname, dpi=400)
    fig.savefig(output_fname, dpi=400)
    plt.close(fig)



def plot_median_spectra_with_confidence_intervals(
    wl_grid: np.ndarray,
    posterior_median_spectra: np.ndarray,
    posterior_bounds_spectra: np.ndarray,
    real_spectra: np.ndarray,
    ideal_spectra: np.ndarray,
    noises: np.ndarray,
    colors: List[str],
    model_labels: List[str],
    output_dir: str
):

    n_samples = len(real_spectra)

    for idx in tqdm(range(n_samples), desc="Plotting spectra..."):
        real_spectrum = real_spectra[idx]
        ideal_spectrum = ideal_spectra[idx]
        noise = noises[idx]
        
        od = os.path.join(output_dir, f"sample_{idx}")
        if not Path(od).exists():
            Path(od).mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.grid(alpha=0.5)

        ax.plot(wl_grid, real_spectrum, color="black", linestyle="--", label="real")
        ax = plot_spectrum_with_confidence_interval(
            ax, wl_grid, ideal_spectrum, noise, alphas=[0.5, 0.2], color="green", label="ideal", 
        )

        for model_label, posterior_median_spectrum, posterior_bounds_spectrum, color in zip(
            model_labels, posterior_median_spectra, posterior_bounds_spectra, colors
        ):
            ax = plot_spectrum_with_confidence_interval(
                ax, wl_grid, posterior_median_spectrum[idx], posterior_bounds_spectrum[idx], alphas=[0.5, 0.2], color=color, label=model_label
            )

        # ax.legend(fontsize='large')
        # fig.legend(loc='outside upper center', fontsize='large', bbox_to_anchor=(0.5, 1.05), ncol=3)
        handles = [Line2D([0], [0], lw=1.0, ls="--", color="black"),
                   Line2D([0], [0], lw=1.0, ls="--", color="green"),
        ]
        handles += [
            Line2D([0], [0], color=c, lw=4)
            for c in colors
        ]
        fig.legend(
            handles=handles,
            labels=[rf'Real spectrum ($\tilde{{x}}_{{{idx}}}$)', rf'Ideal spectrum ($x_{{{idx}}}$)'] + [label for label in model_labels],
            ncols=5,
            frameon=False,
            loc="outside upper center",
            fontsize=12,
        )
        fig.tight_layout()
        fig.subplots_adjust(top=0.85, 
                            bottom=0.15, 
                            left=0.1,
                            right=0.9,
                            hspace=0.2,
                            wspace=0.2)
        ax.set_ylabel(r"Transit depth [$(\mathrm{R_p}/\mathrm{R_*})^2$]")
        ax.set_xlim(np.min(wl_grid) - 0.05 * np.min(wl_grid), np.max(wl_grid) + 0.05 * np.max(wl_grid))
        ax.set_xlabel(r"Wavelength [$\mathrm{\mu m}$]")

        if np.max(wl_grid) - np.min(wl_grid) > 5:
            ax.set_xscale("log")
            ax.tick_params(axis="x", which="minor")
        
        ## multiple by 100 to turn it into percentage. 
        # spectrum[idx, 0, :] => wl_grid
        # spectrum[idx, 1, :] => observed spectrum
        # spectrum[idx, 2, :] => noise
        # plt.errorbar(x=spectrum[:,0], y= spectrum[:,1]*100, yerr=spectrum[:,2]*100 )
        # output_fname = os.path.join(od, f"spectrum_{idx}.png")
        output_fname = os.path.join(od, f"spectrum_{idx}")
        fig.savefig(f"{output_fname}.png", dpi=400)
        fig.savefig(f"{output_fname}.pdf", format='pdf', bbox_inches='tight', dpi=400)
        plt.close(fig)
    

def visualise_spectrum(spectrum):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(10,6))
    plt.errorbar(x=spectrum[:,0], y= spectrum[:,1], yerr=spectrum[:,2] )
    ## usually we visualise it in log-scale
    plt.xscale('log')
    plt.show()
