import numpy as np
from scipy import stats
import sys
sys.path.append("../")

from utils.prior import restrict_to_prior
from utils.resampling import resample_equal, normalise_arr
from utils.fm import compute_approx_mean_and_bound, check_output
from utils import timeit
from typing import Callable

# avoid to use this function
def preprocess_trace_for_posterior_loss(tr, weights, bounds):
    trace_resampled = resample_equal(tr,weights )
    trace = normalise_arr(trace_resampled, bounds )
    return trace 


def L2_loss(truth, predicted):
    """Simple MSE"""
    return np.mean(np.square(truth-predicted))
def L1_loss(truth, predicted):
    """Simple MAE"""
    return np.mean(np.abs(truth-predicted))

def huber_loss(truth, predicted, alpha):
    """huber loss with threshold (alpha) set at 1"""
    if alpha >= 1:  
        return L2_loss(truth, predicted)
    else:
        return L1_loss(truth, predicted)

def compute_posterior_loss(
    posterior_samples: np.ndarray, 
    reference_samples: np.ndarray, 
    bounds_matrix: np.ndarray, 
    restrict: bool = True, 
    normalize: bool = True
    ):
    n_repeats, n_targets = posterior_samples.shape
    # trace1 = preprocess_trace_for_posterior_loss(tr1, weight1, bounds_matrix)
    # trace2 = preprocess_trace_for_posterior_loss(tr2, weight2, bounds_matrix)

    # normalize array according to prior bounds
    if normalize:
        posterior_samples = normalise_arr(posterior_samples, bounds_matrix, restrict)
        reference_samples = normalise_arr(reference_samples, bounds_matrix, restrict)

    posterior_score = []
    for t in range(0, n_targets):

        metric_ks = stats.ks_2samp(posterior_samples[:,t], reference_samples[:,t])
        posterior_score.append((1 - metric_ks.statistic) * 1000)
    
    # print(f"Score trace: {score_trace}")
    return np.array(posterior_score) 


@timeit
def compute_spectral_score(median, bound, GT_median, GT_bound):
    """compute the score contribution from the similaries between two spectra.

    Args:
        median (array): median spectra from participants
        bound (array): The IQR bound from participants. 
        GT_median (array): median spectra generated from GT
        GT_bound (array): The IQR bound from GT.

    Returns:
        scalar: the score from spectral loss
    """
    GT_level = np.mean(GT_median)
    level = np.mean(median)
    alpha = np.abs(np.log10(level/GT_level))
    log_truth = np.log10(GT_median)
    log_predicted = np.log10(median)
    median_loss = 100*huber_loss(log_truth,log_predicted,alpha)
    log_bound = np.log10(bound)
    log_GTbound = np.log10(GT_bound)
    mean_bound = np.mean(bound)
    mean_GTbound = np.mean(GT_bound) 
    alpha_bound = np.abs(np.log10(mean_bound/mean_GTbound))
    bound_loss = 100*huber_loss(log_GTbound, log_bound,alpha_bound)
    score = 1000-np.mean([bound_loss,median_loss])
    ## the minimum score is 0 
    score = np.maximum(score, 0)
    return score


def compute_median_and_bound(
    posterior_samples: np.ndarray, 
    weights: np.ndarray, 
    bounds_matrix: np.ndarray,
    fm_func: Callable,
    q_list: np.ndarray
    ):

    tr1 = restrict_to_prior(posterior_samples, bounds_matrix)
    q1, q2, q3 = compute_approx_mean_and_bound(tr1, weights, fm_func, q_list)
    q1, q2, q3 = check_output(q1, q2, q3)
    median, bound = q2, q3 - q1 + 1e-8
    return median, bound


def compute_spectral_loss(
    posterior_samples: np.ndarray, 
    reference_samples: np.ndarray, 
    weights: np.ndarray, 
    bounds_matrix: np.ndarray, 
    fm_func: Callable, 
    q_list: np.ndarray
    ):
    posterior_median, posterior_bound = compute_median_and_bound(posterior_samples, weights, bounds_matrix, fm_func, q_list)
    reference_median, reference_bound = compute_median_and_bound(reference_samples, weights, bounds_matrix, fm_func, q_list)


    score = compute_spectral_score(
        posterior_median, posterior_bound, reference_median, reference_bound
    )
    return [score, posterior_median, posterior_bound, reference_median, reference_bound]