import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
import corner
from sbi.analysis import pairplot
from sbi.analysis.plot import sbc_rank_plot
from sbi.analysis.plot import plot_tarp
from metrics.sbc import run_sbc, check_sbc
from metrics.tarp import run_tarp, check_tarp
from plotter.corners import corner_plot_prior_posterior


from netcal.metrics.regression import NLL, PinballLoss, PICP, QCE, ENCE, UCE
from netcal.presentation import ReliabilityRegression, ReliabilityQCE
# from plotter.diagrams import plot_regression_diagrams
import argparse
from pathlib import Path
import yaml
import os
from os.path import join

import torch
import numpy as np
import torch


def perform_sbc_evaluation(
    posterior_samples: torch.Tensor,
    thetas: torch.Tensor,
    model_label: str
    ):

    sbc_output_dir = os.path.join(output_dir, "sbc")
    if not Path(sbc_output_dir).exists():
        Path(sbc_output_dir).mkdir(parents=True, exist_ok=True)
    

    num_samples, num_posterior_samples, dim_theta = posterior_samples.shape
    ranks, dap_samples = run_sbc(
        thetas,
        posterior_samples,
    )

    f, ax = sbc_rank_plot(
        ranks=ranks,
        num_posterior_samples=num_posterior_samples,
        plot_type="hist",
        num_bins=None,  # by passing None we use a heuristic for the number of bins.
    )
    f.savefig(join(sbc_output_dir, "hist.png"), dpi=400)

    f, ax = sbc_rank_plot(ranks, num_posterior_samples, plot_type="cdf")
    f.savefig(join(sbc_output_dir, "cdf.png"), dpi=400)

    check_stats = check_sbc(ranks, thetas, dap_samples, num_posterior_samples)
    print(check_stats)

    with open(join(sbc_output_dir, "stats.pkl"), "wb") as f:
        pickle.dump(check_stats, f)

    return ranks, dap_samples, check_stats



def perform_tarp_evaluation(
    posterior_samples: torch.Tensor,
    thetas: torch.Tensor,
    model_label: str
):
    tarp_output_dir = os.path.join(output_dir, "tarp")
    if not Path(tarp_output_dir).exists():
        Path(tarp_output_dir).mkdir(parents=True, exist_ok=True)
    

    # tarp expects (n_posterior_samples, n_tarp_samples, n_dims) for posterior_samples
    posterior_samples = posterior_samples.permute(1, 0, 2)
    
    ecp, alpha = run_tarp(
        thetas,
        posterior_samples,
        references=None,  # will be calculated automatically.
    )

    # Similar to SBC, we can check then check whether the distribution of ecp is close to
    # that of alpha.
    atc, ks_pval = check_tarp(ecp, alpha)
    print(atc, "Should be close to 0")
    print(ks_pval, "Should be larger than 0.05")

    # Or, we can perform a visual check.
    fig, ax = plot_tarp(ecp, alpha)
    fig.savefig(join(tarp_output_dir, "ecp.png"), dpi=400)
    plt.close(fig)

    tarp_results = {
        "ecp": ecp,
        "alpha": alpha,
        "atc": atc,
        "ks_pval": ks_pval,
    }

    
    with open(join(tarp_output_dir, "stats.pkl"), "wb") as f:
        pickle.dump(tarp_results, f)

    return tarp_results


def evaluate_calibration_metrics(
    posterior_samples: np.ndarray,
    thetas: np.ndarray,
    model_label: str,
    output_fname: str
):

    # net:cal regression calibration metrics
    # mean          # NumPy n-D array holding the estimated mean of shape (n, d) with n samples and d dimensions
    # stddev        # NumPy n-D array holding the estimated stddev (independent) of shape (n, d) with n samples and d dimensions
    # ground_truth  # NumPy n-D array holding the ground-truth scores of shape (n, d) with n samples and d dimensions
    
    posterior_mean = np.mean(posterior_samples, axis=1)
    posterior_std = np.std(posterior_samples, axis=1)
    posterior_cov = np.array([np.cov(realizations, rowvar=False) for realizations in posterior_samples])

    univariate = (posterior_mean, posterior_std)
    multivariate = (posterior_mean, posterior_cov)


    print(f"Mean {model_label}: {posterior_mean.shape}")
    print(f"Stddev {model_label}: {posterior_std.shape}")
    print(f"Cov {model_label}: {posterior_cov.shape}")
    print(f"Ground truth: {thetas.shape}")

    # define the quantile levels that are used to evaluate the pinball loss and the QCE
    bins = 20  # used for evaluation metrics
    quantiles = np.linspace(0.05, 0.95, 19)

    # regression 
    # Metrics for Regression Uncertainty Calibration
    # Methods for measuring miscalibration in the context of regression uncertainty calibration for probabilistic
    # regression models.

    # The common methods for regression uncertainty evaluation are *netcal.metrics.regression.PinballLoss* (Pinball
    # loss), the *netcal.metrics.regression.NLL* (NLL), and the *netcal.metrics.regression.QCE* (M-QCE and
    # C-QCE). The Pinball loss as well as the Marginal/Conditional Quantile Calibration Error (M-QCE and C-QCE) evaluate
    # the quality of the estimated quantiles compared to the observed ground-truth quantile coverage. The NLL is a proper
    # scoring rule to measure the overall quality of the predicted probability distributions.

    # Further metrics are the *netcal.metrics.regression.UCE* (UCE) and the *netcal.metrics.regression.ENCE*
    # (ENCE) which both apply a binning scheme over the predicted standard deviation/variance and test for *variance
    # calibration*.

    # For a detailed description of the available metrics within regression calibration, see the module doc of
    # *netcal.regression*.


    nll_loss = NLL()
    # Negative log likelihood (NLL) for probabilistic regression models.
    # If a probabilistic forecaster outputs a probability density function (PDF) :math:`f_Y(y)` targeting the ground-truth
    # :math:`y`, the negative log likelihood is defined by

    # .. math::
    #     \\text{NLL} = -\\sum^N_{n=1} \\log\\big(f_Y(y)\\big) ,

    # with :math:`N` as the number of samples within the examined data set.

    # **Note:** the input field for the standard deviation might also be given as quadratic NumPy arrays of shape
    # (n, d, d) with d dimensions. In this case, this method asserts covariance matrices as input
    # for each sample and the NLL is calculated for multivariate distributions.


    pinball_loss = PinballLoss()
    # Pinball aka quantile loss within regression calibration to test for *quantile calibration* of a probabilistic
    # regression model. The Pinball loss is an asymmetric loss that measures the quality of the predicted
    # quantiles. Given a probabilistic regression model that outputs a probability density function (PDF) :math:`f_Y(y)`
    # targeting the ground-truth :math:`y`, we further denote the cumulative as :math:`F_Y(y)` and the (inverse)
    # percent point function (PPF) as :math:`F_Y^{-1}(\\tau)` for a certain quantile level :math:`\\tau \\in [0, 1]`.

    # The Pinball loss is given by

    # .. math::
    #    L_{\\text{Pin}}(\\tau) =
    #    \\begin{cases}
    #         \\big( y-F_Y^{-1}(\\tau) \\big)\\tau \\quad &\\text{if } y \\geq F_Y^{-1}(\\tau)\\\\
    #         \\big( F_Y^{-1}(\\tau)-y \\big)(1-\\tau) \\quad &\\text{if } y < F_Y^{-1}(\\tau)
    #    \\end{cases} .
    # """



    qce_loss = QCE(bins=bins, marginal=True)  # if "marginal=False", we can also measure the QCE by means of the predicted variance levels (realized by binning the variance space)
    # Marginal Quantile Calibration Error (M-QCE) and Conditional Quantile Calibration Error (C-QCE) which both measure
    # the gap between predicted quantiles and observed quantile coverage also for multivariate distributions.
    # The M-QCE and C-QCE have originally been proposed by [1]_.
    # The derivation of both metrics are based on
    # the Normalized Estimation Error Squared (NEES) known from object tracking [2]_.
    # The derivation of both metrics is shown in the following.

    # **Definition of standard NEES:**
    # Given mean prediction :math:`\\hat{\\boldsymbol{y}} \\in \\mathbb{R}^M`, ground-truth
    # :math:`\\boldsymbol{y} \\in \\mathbb{R}^M`, and
    # estimated covariance matrix :math:`\\hat{\\boldsymbol{\\Sigma}} \\in \\mathbb{R}^{M \\times M}` using
    # :math:`M` dimensions, the NEES is defined as

    # .. math::
    #     \\epsilon = (\\boldsymbol{y} - \\hat{\\boldsymbol{y}})^\\top \\hat{\\boldsymbol{\\Sigma}}^{-1}
    #     (\\boldsymbol{y} - \\hat{\\boldsymbol{y}}) .

    # The average NEES is defined as the mean error over :math:`N` trials in a Monte-Carlo simulation for
    # Kalman-Filter testing, so that

    # .. math::
    #     \\bar{\\epsilon} = \\frac{1}{N} \\sum^N_{i=1} \\epsilon_i .

    # Under the condition, that :math:`\\mathbb{E}[\\boldsymbol{y} - \\hat{\\boldsymbol{y}}] = \\boldsymbol{0}` (zero mean),
    # a :math:`\\chi^2`-test is performed to evaluate the estimated uncertainty. This test is accepted, if

    # .. math::
    #     \\bar{\\epsilon} \\leq \\chi^2_M(\\tau),

    # where :math:`\\chi^2_M(\\tau)` is the PPF score obtained by a :math:`\\chi^2` distribution
    # with :math:`M` degrees of freedom, for a certain quantile level :math:`\\tau \\in [0, 1]`.

    # **Marginal Quantile Calibration Error (M-QCE):**
    # In the case of regression calibration testing, we are interested in the gap between predicted quantile levels and
    # observed quantile coverage probability for a certain set of quantile levels. We assert :math:`N` observations of our
    # test set that are used to estimate the NEES, so that we can compute the expected deviation between predicted
    # quantile level and observed quantile coverage by

    # .. math::
    #     \\text{M-QCE}(\\tau) := \\mathbb{E} \\Big[ \\big| \\mathbb{P} \\big( \\epsilon \\leq \\chi^2_M(\\tau) \\big) - \\tau \\big| \\Big] ,

    # which is the definition of the Marginal Quantile Calibration Error (M-QCE) [1]_.
    # The M-QCE is calculated by

    # .. math::
    #     \\text{M-QCE}(\\tau) = \\Bigg| \\frac{1}{N} \\sum^N_{n=1} \\mathbb{1} \\big( \\epsilon_n \\leq \\chi^2_M(\\tau) \\big) - \\tau \\Bigg|

    # **Conditional Quantile Calibration Error (C-QCE):**
    # The M-QCE measures the marginal calibration error which is more suitable to test for *quantile calibration*.
    # However, similar to :class:`netcal.metrics.regression.UCE` and :class:`netcal.metrics.regression.ENCE`,
    # we want to induce a dependency on the estimated covariance, since we require
    # that

    # .. math::
    #     &\\mathbb{E}[(\\boldsymbol{y} - \\hat{\\boldsymbol{y}})(\\boldsymbol{y} - \\hat{\\boldsymbol{y}})^\\top |
    #     \\hat{\\boldsymbol{\\Sigma}} = \\boldsymbol{\\Sigma}] = \\boldsymbol{\\Sigma},

    #     &\\forall \\boldsymbol{\\Sigma} \\in \\mathbb{R}^{M \\times M}, \\boldsymbol{\\Sigma} \\succcurlyeq 0,
    #     \\boldsymbol{\\Sigma}^\\top = \\boldsymbol{\\Sigma} .

    # To estimate the a *covariance* dependent QCE, we apply a binning scheme (similar to the
    # :class:`netcal.metrics.confidence.ECE`) over the square root of the *standardized generalized variance* (SGV) [3]_,
    # that is defined as

    # .. math::
    #     \\sigma_G = \\sqrt{\\text{det}(\\hat{\\boldsymbol{\\Sigma}})^{\\frac{1}{M}}} .

    # Using the generalized standard deviation, it is possible to get a summarized statistic across different
    # combinations of correlations to denote the distribution's dispersion. Thus, the Conditional Quantile Calibration
    # Error (C-QCE) [1]_ is defined by

    # .. math::
    #     \\text{C-QCE}(\\tau) := \\mathbb{E}_{\\sigma_G, X}\\Big[\\Big|\\mathbb{P}\\big(\\epsilon \\leq \\chi^2_M(\\tau) | \\sigma_G\\big) - \\tau \\Big|\\Big] ,

    # To approximate the expectation over the generalized standard deviation, we use a binning scheme with :math:`B` bins
    # (similar to the ECE) and :math:`N_b` samples per bin to compute the weighted sum across all bins, so that

    # .. math::
    #     \\text{C-QCE}(\\tau) \\approx \\sum^B_{b=1} \\frac{N_b}{N} | \\text{freq}(b) - \\tau |

    # where :math:`\\text{freq}(b)` is the coverage frequency within bin :math:`b` and given by

    # .. math::
    #     \\text{freq}(b) = \\frac{1}{N_b} \\sum_{n \\in \\mathcal{M}_b} \\mathbb{1}\\big(\\epsilon_i \\leq \\chi^2_M(\\tau)\\big) ,

    # with :math:`\\mathcal{M}_b` as the set of indices within bin :math:`b`.
    
    picp_loss = PICP(bins=bins)
    # Compute Prediction Interval Coverage Probability (PICP) and Mean Prediction Interval Width (MPIW).
    # These metrics have been proposed by [1]_, [2]_.
    # This metric is used for Bayesian models to determine the quality of the uncertainty estimates.
    # In Bayesian mode, an uncertainty estimate is attached to each sample. The PICP measures the probability, that
    # the true (observed) accuracy falls into the p% prediction interval. The uncertainty is well-calibrated, if
    # the PICP is equal to p%. Simultaneously, the MPIW measures the mean width of all prediction intervals to evaluate
    # the sharpness of the uncertainty estimates.
    
    ence_loss = ENCE(bins=bins)
    # Expected Normalized Calibration Error (ENCE) for a regression calibration evaluation to test for
    # *variance calibration*. A probabilistic regression model takes :math:`X` as input and outputs a
    # mean :math:`\\mu_Y(X)` and a standard deviation :math:`\\sigma_Y(X)` targeting the ground-truth :math:`y`.
    # Similar to the :class:`netcal.metrics.confidence.ECE`, the ENCE applies a binning scheme with :math:`B` bins
    # over the predicted standard deviation :math:`\\sigma_Y(X)` and measures the absolute (normalized) difference
    # between root mean squared error (RMSE) and root mean variance (RMV) [1]_.
    # Thus, the ENCE [1]_ is defined by

    # .. math::
    #     \\text{ENCE} := \\frac{1}{B} \\sum^B_{b=1} \\frac{|RMSE(b) - RMV(b)|}{RMV(b)} ,

    # where :math:`RMSE(b)` and :math:`RMV(b)` are the root mean squared error and the root mean variance within
    # bin :math:`b`, respectively.

    # If multiple dimensions are given, the ENCE is measured for each dimension separately.

    uce_loss = UCE(bins=bins)
    # Uncertainty Calibration Error (UCE) for a regression calibration evaluation to test for
    # *variance calibration*. A probabilistic regression model takes :math:`X` as input and outputs a
    # mean :math:`\\mu_Y(X)` and a variance :math:`\\sigma_Y^2(X)` targeting the ground-truth :math:`y`.
    # Similar to the :class:`netcal.metrics.confidence.ECE`, the UCE applies a binning scheme with :math:`B` bins
    # over the predicted variance :math:`\\sigma_Y^2(X)` and measures the absolute difference
    # between mean squared error (MSE) and mean variance (RMV) [1]_.
    # Thus, the UCE [1]_ is defined by

    # .. math::
    #     \\text{UCE} := \\sum^B_{b=1} \\frac{N_b}{N} |MSE(b) - MV(b)| ,

    # where :math:`MSE(b)` and :math:`MV(b)` are the mean squared error and the mean variance within
    # bin :math:`b`, respectively, and :math:`N_b` is the number of samples within bin :math:`b`.

    # If multiple dimensions are given, the UCE is measured for each dimension separately.


    # univariate
    pinball_loss_independent_model = np.mean(
        pinball_loss.measure(univariate, thetas, q=quantiles, reduction="none"),  # (q, n, d)
        axis=(0, 1)
    )  # (d,)
    nll_loss_independent_model = nll_loss.measure(univariate, thetas, reduction="batchmean")
    qce_loss_independent_model = qce_loss.measure(univariate, thetas, q=quantiles, reduction="batchmean")
    picp_loss_independent_model = picp_loss.measure(univariate, thetas, q=quantiles, reduction="batchmean")
    
    # only univariate
    ence_loss_model = ence_loss.measure(univariate, thetas)
    uce_loss_model = uce_loss.measure(univariate, thetas)

    # multivariate
    nll_loss_joint_model = nll_loss.measure(multivariate, thetas, reduction="batchmean")  # scalar or (d,)
    qce_loss_joint_model = qce_loss.measure(multivariate, thetas, q=quantiles, reduction="batchmean")

    if nll_loss_joint_model.size > 1:
            nll_loss_joint_model = np.sum(nll_loss_joint_model)

    if isinstance(qce_loss_joint_model, np.ndarray) and qce_loss_joint_model.size > 1:
        qce_loss_joint_model = np.mean(qce_loss_joint_model)


    print("################################")
    print("Regression metrics:")
    print(f"NLL {model_label} (independent): ", nll_loss_independent_model)
    print(f"PINBALL {model_label} (independent): ",  pinball_loss_independent_model)
    print(f"QCE {model_label} (independent): ", qce_loss_independent_model)
    print(f"PICP {model_label} (independent): ", picp_loss_independent_model)
    print(f"ENCE {model_label}: ", ence_loss_model)
    print(f"UCE {model_label}: ", uce_loss_model)
    print(f"ENCE {model_label} (average): ", np.mean(ence_loss_model))
    print(f"UCE {model_label} (average): ", np.mean(uce_loss_model))
    print(f"NLL {model_label} (joint): ", nll_loss_joint_model)
    print(f"QCE {model_label} (joint): ", qce_loss_joint_model)
    print("################################")

    calibration_metrics = {            
        "NLL": {
            "independent": nll_loss_independent_model,
            "avg_independent": np.mean(nll_loss_independent_model),
            "joint": nll_loss_joint_model,
        },
        "QCE": {
            "independent": qce_loss_independent_model,
            "avg_independent": np.mean(qce_loss_independent_model),
            "joint": qce_loss_joint_model,
        },
        "Pinball": pinball_loss_independent_model,
        "PICP": {
            "picp" : picp_loss_independent_model[0],
            "mpiw" : picp_loss_independent_model[1],
        },
        "ENCE": {
            "independent": ence_loss_model,
            "joint": np.mean(ence_loss_model),
        },
        "UCE": {
            "independent": uce_loss_model,
            "joint": np.mean(uce_loss_model),
        },
    }

    with open(output_fname, "wb") as f:
        pickle.dump(calibration_metrics, f)
    return calibration_metrics


def plot_calibration_diagrams(
    posterior_samples: torch.Tensor,
    thetas: torch.Tensor,
    model_label: str,
    output_dir: str
    ):
    # net:cal regression calibration metrics
    # mean          # NumPy n-D array holding the estimated mean of shape (n, d) with n samples and d dimensions
    # stddev        # NumPy n-D array holding the estimated stddev (independent) of shape (n, d) with n samples and d dimensions
    # ground_truth  # NumPy n-D array holding the ground-truth scores of shape (n, d) with n samples and d dimensions

    if not Path(output_dir).exists():
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    
    posterior_mean = np.mean(posterior_samples, axis=1)
    posterior_std = np.std(posterior_samples, axis=1)
    posterior_cov = np.array([np.cov(realizations, rowvar=False) for realizations in posterior_samples])

    univariate = (posterior_mean, posterior_std)
    multivariate = (posterior_mean, posterior_cov)

    # define the quantile levels that are used to evaluate the pinball loss and the QCE
    bins = 20  # used for evaluation metrics
    quantiles = np.linspace(0.05, 0.95, 19)

    # -------------------------------------------------
    # visualization

    # initialize the diagram object
    diagram = ReliabilityRegression(quantiles=bins + 1)
    # Reliability diagram in the scope of regression calibration for probabilistic regression models.
    # This diagram visualizes the quantile coverage frequency for several quantile levels and plots these observed
    # coverage scores above the desired quantile levels.
    # In this way, it is possible to compare the predicted and the observed quantile levels with each other.

    # This method is able to visualize the quantile coverage in terms of multiple univariate distributions if the input
    # is given as multiple independent Gaussians.
    # This method is also able to visualize the multivariate quantile coverage for a joint multivariate Gaussian if the
    # input is given with covariance matrices.

    diagram.plot(univariate, thetas, filename=os.path.join(output_dir, "rr_independent.png")) # independent 
    diagram.plot(multivariate, thetas, filename=os.path.join(output_dir, "rr_joint.png")) # joint



    diagram = ReliabilityQCE(bins=bins)
    # Visualizes the Conditional Quantile Calibration Error (C-QCE) in the scope of regression calibration as a bar chart
    # for probabilistic regression models.
    # See :class:`netcal.metrics.regression.QCE` for a detailed documentation of the C-QCE [1]_.
    # This method is able to visualize the C-QCE in terms of multiple univariate distributions if the input is given
    # as multiple independent Gaussians.
    # This method is also able to visualize the multivariate C-QCE for a multivariate Gaussian if the input is given
    # with covariance matrices.

    diagram.plot(univariate, thetas, q=quantiles, filename=os.path.join(output_dir, "qce_independent.png")) # independent
    diagram.plot(multivariate, thetas, q=quantiles, filename=os.path.join(output_dir, "qce_joint.png")) # joint



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir", required=True, help="Base save directory for the evaluation")
    parser.add_argument("--settings_file", required=True, help="Settings file for the evaluation")
    parser.add_argument("--model_label", required=True, help="Model label")
    parser.add_argument("--latest", action="store_true", default=False, help="Use the latest model in the directory")
    args = parser.parse_args()
    model_label = str(args.model_label)

    with open(args.settings_file, 'r') as f:
        settings = yaml.safe_load(f)

    data_dir = settings['dataset']['path']

    # create dirs
    output_dir = os.path.join(args.eval_dir, "calibration")
    if not Path(output_dir).exists():
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    # load targets
    thetas = torch.tensor(np.load(os.path.join(data_dir, 'test_targets.npy')))

    # load posterior samples
    posterior_samples = torch.tensor(
        np.load(os.path.join(args.eval_dir, 'posterior_distribution.npy'))
    ) # original domain, (n_samples, n_repeats, dim_theta)
    
    print(f"Posterior distribution shape: {posterior_samples.shape}")
    
    # perform one-dimensional sbc ranking
    # multi-dimensional sbc requires log probs of reference samples
    ranks, dap_samples, sbc_stats = perform_sbc_evaluation(posterior_samples, thetas, model_label)

    # Posterior calibration with TARP (Lemos et al. 2023)
    # the tarp method returns the ECP values for a given set of alpha coverage levels.
    tarp_results = perform_tarp_evaluation(posterior_samples, thetas, model_label)
    
    
    ## Regression Calibration Metrics
    # In regression calibration, the most common metric is the Negative Log Likelihood (NLL) to measure the quality of a predicted probability distribution w.r.t. the ground-truth:
    # Negative Log Likelihood (NLL) (netcal.metrics.NLL)
    # The metrics Pinball Loss, Prediction Interval Coverage Probability (PICP), 
    # and Quantile Calibration Error (QCE) evaluate the estimated distributions by means of the predicted quantiles. 
    # For example, if a forecaster makes 100 predictions using a probability distribution for each estimate targeting the true ground-truth, 
    # we can measure the coverage of the ground-truth samples for a certain quantile level (e.g., 95% quantile). 
    # If the relative amount of ground-truth samples falling into a certain predicted quantile is above or below the specified quantile level, 
    # a forecaster is told to be miscalibrated in terms of quantile calibration. Appropriate metrics in this context are

    # Pinball Loss (netcal.metrics.PinballLoss)
    # Prediction Interval Coverage Probability (PICP) [14] (netcal.metrics.PICP)
    # Quantile Calibration Error (QCE) [15] (netcal.metrics.QCE)
    # Finally, if we work with normal distributions, we can measure the quality of the predicted variance/stddev estimates. 
    # For variance calibration, it is required that the predicted variance matches the observed error variance which is equivalent to then Mean Squared Error (MSE). 
    # Metrics for variance calibration are

    # Expected Normalized Calibration Error (ENCE) [17] (netcal.metrics.ENCE)
    # Uncertainty Calibration Error (UCE) [18] (netcal.metrics.UCE)

    # Calibration plot
    # 1. Compute histogram of observed thetas
    # 2. Compute histogram of posterior thetas
    # 3. For each confidence level, compute the fraction of posterior samples that fall within the confidence interval.



    # net:cal regression calibration metrics
    # mean          # NumPy n-D array holding the estimated mean of shape (n, d) with n samples and d dimensions
    # stddev        # NumPy n-D array holding the estimated stddev (independent) of shape (n, d) with n samples and d dimensions
    # ground_truth  # NumPy n-D array holding the ground-truth scores of shape (n, d) with n samples and d dimensions
    # https://github.com/EFS-OpenSource/calibration-framework/blob/main/examples/regression/multivariate/main.py
    
    posterior_samples = posterior_samples.numpy()
    thetas = thetas.numpy()


    regression_metrics_output_dir = os.path.join(output_dir, "regression")
    if not Path(regression_metrics_output_dir).exists():
        Path(regression_metrics_output_dir).mkdir(parents=True, exist_ok=True)


    results = evaluate_calibration_metrics(
        posterior_samples, 
        thetas, 
        model_label,
        os.path.join(regression_metrics_output_dir, "metrics.pkl")
    )
    # results["sbc"] = sbc_stats
    # results["tarp"] = tarp_results


    plot_calibration_diagrams(
        posterior_samples, 
        thetas, 
        model_label,
        os.path.join(output_dir, "diagrams")
    )

    # plot corner plot
    corner_plot_prior_posterior(
        posterior_samples, 
        thetas, 
        model_label, 
        os.path.join(output_dir, "diagrams", "prior_vs_posteriors.png")
    )

    