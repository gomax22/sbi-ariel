import numpy as np
import matplotlib.pyplot as plt

def plot_histogram_traces(
    posterior_samples: np.ndarray, 
    reference_samples: np.ndarray, 
    output_fname: str):

    n_targets = posterior_samples.shape[-1]

    # trace_model, trace_ns same domain
    fig, ax = plt.subplots(1, n_targets, figsize=(25, 5))
    
    # create bins for each target parameter
    for n_target in range(n_targets):
        bins = np.linspace(min(posterior_samples[:, n_target].min(), reference_samples[:, n_target].min()), max(posterior_samples[:, n_target].max(), reference_samples[:, n_target].max()), 100)
        ax[n_target].hist(posterior_samples[:, n_target], bins, alpha=0.5, label='trace1')
        ax[n_target].hist(reference_samples[:, n_target], bins, alpha=0.5, label='trace2')
        ax[n_target].legend(loc='upper right',  fontsize="small")
        ax[n_target].set_title(f"tr1_vs_rs ({len(posterior_samples)},{len(reference_samples)})", fontsize="small")    
                
    fig.savefig(output_fname, dpi=400)
    plt.close(fig)