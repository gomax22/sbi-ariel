import numpy as np
import os
import matplotlib.pyplot as plt
from netcal.presentation import ReliabilityQCE, ReliabilityRegression
from pathlib import Path
from typing import List, Tuple, Union
import matplotlib as mpl
from matplotlib.lines import Line2D

from netcal import is_in_quantile
import tikzplotlib
mpl.rcParams['text.usetex'] = False 

# extended from netcal.presentation.ReliabilityRegression
class RelibilityRegressions(ReliabilityRegression):
    def __init__(self, quantiles = 11):
        super().__init__(quantiles)

    def plots(
        self,
        X: Union[List[Tuple[np.ndarray, np.ndarray]], List[np.ndarray]],
        y: np.ndarray,
        colors: List[str],
        model_labels: List[str],
        *,
        kind: str = 'meanstd',
        filename: str = None,
        tikz: bool = False,
        title_suffix: str = None,
        feature_names: List[str] = None,
        fig: plt.Figure = None,
        **save_args
    ) -> Union[plt.Figure, str]:
        """
        Reliability diagram for regression calibration to visualize the predicted quantile levels vs. the actually
        observed quantile coverage probability.
        This method is able to visualize the reliability diagram in terms of multiple univariate distributions if the
        input is given as multiple independent Gaussians.
        This method is also able to visualize the joint multivariate quantile calibration for a multivariate Gaussian
        if the input is given with covariance matrices (see parameter "kind" for a detailed description of the input
        format).

        Parameters
        ----------
        X : np.ndarray of shape (r, n, [d]) or (t, n, [d]), or Tuple of two np.ndarray, each of shape (n, [d])
            Input data obtained by a model that performs inference with uncertainty.
            See parameter "kind" for input format descriptions.
        y : np.ndarray of shape (n, [d])
            Target scores for each prediction estimate in X.
        kind : str, either "meanstd" or "cumulative"
            Specify the kind of the input data. Might be one of:
            - meanstd: if X is tuple of two NumPy arrays with shape (n, [d]) and (n, [d, [d]]), this method asserts the
                        first array as mean and the second one as the according stddev predictions for d dimensions.
                        If the second NumPy array has shape (n, d, d), this method asserts covariance matrices as input
                        for each sample. In this case, the NLL is calculated for multivariate distributions.
                        If X is single NumPy array of shape (r, n), this methods asserts predictions obtained by a stochastic
                        inference model (e.g. network using MC dropout) with n samples and r stochastic forward passes. In this
                        case, the mean and stddev is computed automatically.
            - cumulative: assert X as tuple of two NumPy arrays of shape (t, n, [d]) with t points on the cumulative
                            for sample n (and optionally d dimensions).
        filename : str, optional, default: None
            Optional filename to save the plotted figure.
        tikz : bool, optional, default: False
            If True, use 'tikzplotlib' package to return tikz-code for Latex rather than a Matplotlib figure.
        title_suffix : str, optional, default: None
            Suffix for plot title.
        feature_names : list, optional, default: None
            Names of the additional features that are attached to the axes of a reliability diagram.
        fig: plt.Figure, optional, default: None
            If given, the figure instance is used to draw the reliability diagram.
            If fig is None, a new one will be created.
        **save_args : args
            Additional arguments passed to 'matplotlib.pyplot.Figure.savefig' function if 'tikz' is False.
            If 'tikz' is True, the argument are passed to 'tikzplotlib.get_tikz_code' function.

        Returns
        -------
        matplotlib.pyplot.Figure if 'tikz' is False else str with tikz code.
            Visualization of the quantile calibration either as Matplotlib figure or as string with tikz code.
        """

        assert kind in ['meanstd', 'cauchy', 'cumulative'], 'Parameter \'kind\' must be either \'meanstd\', or \'cumulative\'.'

        # get quantile coverage of input
        in_quantiles = []
        for x in X:
            in_quantile, _, _, _, _ = is_in_quantile(x, y, self.quantiles, kind) # (q, n, [d]), (q, n, d), (n, d), (n, d, [d])
            in_quantiles.append(in_quantile)


        # get the frequency of which y is within the quantile bounds
        frequencies = []
        for in_quantile in in_quantiles:
            frequency = np.mean(in_quantile, axis=1) # (q, [d])
            frequencies.append(frequency)

        # make frequency array at least 2d
        for idx, frequency in enumerate(frequencies):
            if frequency.ndim == 1:
                frequencies[idx] = np.expand_dims(frequency, axis=1)  # (q, d) or (q, 1)

        n_dims = frequencies[0].shape[-1]

        # check feature names parameter
        if feature_names is not None:
            assert isinstance(feature_names, (list, tuple)), "Parameter \'feature_names\' must be tuple or list."
            assert len(feature_names) == n_dims, "Length of parameter \'feature_names\' must be equal to the amount " \
                                                    "of dimensions. Input with full covariance matrices is interpreted " \
                                                    "as n_features=1."

        # initialize plot and create an own chart for each dimension
        if fig is None:
            fig, axes = plt.subplots(nrows=n_dims, figsize=(7, 3 * n_dims), squeeze=False)
        else:
            axes = [fig.add_subplot(n_dims, 1, idx) for idx in range(1, n_dims + 1)]

        for dim, ax in enumerate(axes):

            # ax object also has an extra dim for columns
            ax = ax[0]

            for frequency, color, model_label in zip(frequencies, colors, model_labels):
                ax.plot(self.quantiles, frequency[:, dim], "o-", color=color, label=model_label, alpha=0.5)

            # draw diagonal as perfect calibration line
            ax.plot([0, 1], [0, 1], color='black', linestyle='--', label='Perfect Calibration')
            ax.set_xlim((0.0, 1.0))
            ax.set_ylim((0.0, 1.0))

            # labels and legend of second plot
            ax.set_xlabel('Expected quantile')
            ax.set_ylabel('Observed frequency')
            # ax.legend(model_labels + ['Perfect Calibration'], loc='upper left')
            handles = [Line2D([0], [0], lw=1.0, ls="--", color="black")]
            handles += [
                Line2D([0], [0], color=c, lw=1.0)
                for c in colors
            ]
            ax.legend(
                handles=handles,
                labels=["Perfect Calibration"] + [label for label in model_labels],
                ncols=1,
                # frameon=False,
                loc="center right",
                fontsize=10,
                bbox_to_anchor=(1.5, 0.5),
            )
            
            ax.grid(True)

            # set axis title
            title = 'Reliability Regression Diagram'
            if title_suffix is not None:
                title = title + ' - ' + title_suffix
            if feature_names is not None:
                title = title + ' - ' + feature_names[dim]
            else:
                title = title + ' - dim %02d' % dim

            # ax.set_title(title)
        fig.tight_layout()

        # if tikz is true, create tikz code from matplotlib figure
        if tikz:

            # get tikz code for our specific figure and also pass filename to store possible bitmaps
            tikz_fig = tikzplotlib.get_tikz_code(fig, filepath=filename, **save_args)

            # close matplotlib figure when tikz figure is requested to save memory
            plt.close(fig)
            fig = tikz_fig

        # save figure either as matplotlib PNG or as tikz output file
        if filename is not None:
            if tikz:
                with open(filename, "w") as open_file:
                    open_file.write(fig)
            else:
                fig.savefig(filename, **save_args)

        return fig





def plot_reliability_regression_diagrams(
    univariates: List[Tuple[np.ndarray, np.ndarray]],
    multivariates: List[Tuple[np.ndarray, np.ndarray]],
    thetas: np.ndarray,
    labels: List[str],
    colors: List[str],
    model_labels: str,
    output_dir: str,
    bins: int = 20,
    ):

    if not Path(output_dir).exists():
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    num_targets = thetas.shape[-1]

    # -------------------------------------------------
    # visualization
     # initialize the diagram object
    diagram = RelibilityRegressions(quantiles=bins + 1)
    # Reliability diagram in the scope of regression calibration for probabilistic regression models.
    # This diagram visualizes the quantile coverage frequency for several quantile levels and plots these observed
    # coverage scores above the desired quantile levels.
    # In this way, it is possible to compare the predicted and the observed quantile levels with each other.

    # This method is able to visualize the quantile coverage in terms of multiple univariate distributions if the input
    # is given as multiple independent Gaussians.
    # This method is also able to visualize the multivariate quantile coverage for a joint multivariate Gaussian if the
    # input is given with covariance matrices.
    fig = diagram.plots(univariates, thetas, feature_names=labels, colors=colors, model_labels=model_labels)
    # ax = plt.gca()
    # ax.legend(model_labels, loc='upper left')
    
    fig.savefig(os.path.join(output_dir, "rr_independent.png"), bbox_inches='tight', dpi=400)
    fig.savefig(os.path.join(output_dir, "rr_independent.pdf"), format='pdf', bbox_inches='tight', dpi=400)
    plt.close(fig)

    fig = diagram.plots(multivariates, thetas, colors=colors, model_labels=model_labels, feature_names=["Aggregated dimensions"])
    # ax = plt.gca()
    # ax.legend(model_labels, loc='upper left')
    fig.savefig(os.path.join(output_dir, "rr_joint.png"), bbox_inches='tight', dpi=400)
    fig.savefig(os.path.join(output_dir, "rr_joint.pdf"), format='pdf', bbox_inches='tight', dpi=400)
    plt.close(fig)



def plot_qce_diagrams(
    univariates: List[Tuple[np.ndarray, np.ndarray]],
    multivariates: List[Tuple[np.ndarray, np.ndarray]],
    thetas: np.ndarray,
    labels: List[str],
    colors: List[str],
    model_labels: str,
    output_dir: str,
    bins: int = 20,
    quantiles: np.ndarray = np.linspace(0.05, 0.95, 19) 
):
    
    if not Path(output_dir).exists():
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    num_targets = thetas.shape[-1]

    diagram = ReliabilityQCE(bins=bins)
    # Visualizes the Conditional Quantile Calibration Error (C-QCE) in the scope of regression calibration as a bar chart
    # for probabilistic regression models.
    # See :class:`netcal.metrics.regression.QCE` for a detailed documentation of the C-QCE [1]_.
    # This method is able to visualize the C-QCE in terms of multiple univariate distributions if the input is given
    # as multiple independent Gaussians.
    # This method is also able to visualize the multivariate C-QCE for a multivariate Gaussian if the input is given
    # with covariance matrices.
    fig = diagram.plot(univariates[0], thetas, feature_names=labels, q=quantiles, colors=colors[0])
    for univariate, color in zip(univariates[1:], colors[1:]):
        diagram.plot(univariate, thetas, feature_names=labels, q=quantiles, colors=color, fig=fig)
    
    ax = fig.gca()
    ax.legend(model_labels)
    fig.savefig(os.path.join(output_dir, "qce_independent.png"))
    plt.close(fig)

    fig = diagram.plot(multivariates[0], thetas, feature_names=labels, q=quantiles, colors=colors[0])
    for multivariate, color in zip(multivariates[1:], colors[1:]):
        diagram.plot(multivariate, thetas, feature_names=labels, q=quantiles, colors=color, fig=fig)
    
    ax = fig.gca()
    ax.legend(model_labels)
    fig.savefig(os.path.join(output_dir, "qce_joint.png"))
    plt.close(fig)


def plot_regression_diagrams(
    posteriors: np.ndarray,
    thetas: np.ndarray,
    labels: List[str],
    colors: List[str],
    model_labels: str,
    output_dir: str
    ):

    # net:cal regression calibration metrics
    # mean          # NumPy n-D array holding the estimated mean of shape (n, d) with n samples and d dimensions
    # stddev        # NumPy n-D array holding the estimated stddev (independent) of shape (n, d) with n samples and d dimensions
    # ground_truth  # NumPy n-D array holding the ground-truth scores of shape (n, d) with n samples and d dimensions

    if not Path(output_dir).exists():
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    posterior_means = [np.mean(posterior_samples, axis=1) for posterior_samples in posteriors]
    posterior_stds = [np.std(posterior_samples, axis=1) for posterior_samples in posteriors]
    posterior_covs = [np.array([np.cov(realizations, rowvar=False) for realizations in posterior_samples]) for posterior_samples in posteriors]

    univariates = [(mean, std) for mean, std in zip(posterior_means, posterior_stds)]
    multivariates = [(mean, cov) for mean, cov in zip(posterior_means, posterior_covs)]

    plot_reliability_regression_diagrams(
        univariates=univariates,
        multivariates=multivariates,
        thetas=thetas,
        labels=labels,
        colors=colors,
        model_labels=model_labels,
        output_dir=output_dir
    )

    # plot_qce_diagrams(
    #     univariates=univariates,
    #     multivariates=multivariates,
    #     thetas=thetas,
    #     labels=labels,
    #     colors=colors,
    #     model_labels=model_labels,
    #     output_dir=output_dir
    # )




