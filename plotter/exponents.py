import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from typing import List
from pathlib import Path
import numpy as np
mpl.rcParams['text.usetex'] = False 
mpl.rc('font',family='Times New Roman')

def plot_time_prior_exponents(alphas: List[float], colors: List[str], output_dir: str, eps: float = 0.001):
    """
    Plot the power-law distribution for time prior at different time prior exponents.
    
    Args:
        t (np.ndarray): Time values.
        output_dir (str): Directory to save the plots.
        eps (float): Small value to avoid division by zero.
    """
    t = (1 - eps) * np.linspace(0, 1, 100)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    for exp, color in zip(alphas, colors):
        t_pow = t ** (1 / (1 + exp))
        ax.plot(t, t_pow, color=color, label=rf'$\mathrm{{\alpha}} = {{{exp}}}$')
    ax.plot(t, t, color='black', linestyle='dashed', label=r"$t \sim \mathcal{U}[0,1)$")
    ax.set_xlabel(r"Time ($t$)", fontsize=18)
    ax.set_ylabel(r"$t \propto t^{\frac{1}{1+\alpha}}$", fontsize=18)
    # ax.set_title(r"Power-law distribution for time prior ($p_\alpha (t)$)", fontsize=16)
    ax.grid(True)
    handles = [
        Line2D([0], [0], color=c, lw=4.0)
        for c in colors
    ]
    handles += [Line2D([0], [0], lw=2.0, ls="--", color="black")]
    fig.legend(
        handles=handles,
        labels=[rf'$\mathrm{{\alpha}} = {{{exp}}}$' for exp in alphas] + [r"$t \sim \mathcal{U}[0,1)$"],
        ncols=5,
        frameon=False,
        loc="outside upper center",
        fontsize=14,
    )
    fig.tight_layout()
    fig.subplots_adjust(
        top=0.85,
        bottom=0.15
    )
        
    if not Path(output_dir).exists():
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
    fig.savefig(f"{output_dir}/time_prior_exponents.pdf", format='pdf', bbox_inches='tight', dpi=400)
    plt.close(fig)
    
    
    