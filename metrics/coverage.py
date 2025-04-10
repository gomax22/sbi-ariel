import numpy as np
import torch
from scipy import stats


def compute_confidence_interval(samples: np.ndarray, confidence: float):
    """
    Compute confidence interval for samples
    """
    mean = np.mean(samples, axis=0)
    std = np.std(samples, axis=0, ddof=1)
    z_star = stats.norm.interval(confidence)
    ci_lower, ci_upper = mean + z_star[0] * std, mean + z_star[1] * std
    return ci_lower, ci_upper

def xsigma_coverage_analysis(
    posterior_samples: np.ndarray, 
    theta: np.ndarray,
    confidence: float
    ):

    ci_lower, ci_upper = compute_confidence_interval(posterior_samples, confidence)

    # check whether theta is within x std of the posterior mean
    accuracy = np.logical_and(theta >= ci_lower, theta <= ci_upper).astype(np.float32)
    avg_accuracy = np.mean(accuracy, dtype=np.float32).reshape(1,)
    joint_accuracy = np.all(accuracy).astype(np.float32).reshape(1,)
    # print(f"Accuracy one sigma: {accuracy} ({np.mean(accuracy_one_sigma, dtype=np.float32)})")
    return accuracy, avg_accuracy, joint_accuracy, (ci_lower, ci_upper)

def support_coverage_analysis(
    posterior_samples: np.ndarray, 
    thetas: np.ndarray
    ):

    # check whether theta[i] is within the support of marginal posterior samples
    
    # WARNING: we assume that there are no holes in the support between max and min
    # POSSIBLE SOLUTION: we need to compute the histogram of the samples for each dimension 
    # and check whether the input theta is within a non-zero bin of the histogram
    # this is a more general solution that can be applied to any distribution (but more expensive)
    # Moreover, the number of bins plays a crucial role in the accuracy of the check
    # and the posteriors having a large variace might require a large number of bins 
    # otherwise it will be unfair 
    # ==> normalization could be beneficial in this case, however we would alter the
    # shape of the distribution and the check would be less meaningful
    min_th = np.min(posterior_samples, axis=0)
    max_th = np.max(posterior_samples, axis=0)
    accuracy = np.logical_and(thetas >= min_th, thetas <= max_th).astype(np.float32)
    avg_accuracy = np.mean(accuracy, dtype=np.float32).reshape(1,)
    joint_accuracy = np.all(accuracy).astype(np.float32).reshape(1,)
    return accuracy, avg_accuracy, joint_accuracy


def coverage_analysis(
    posterior_samples: np.ndarray, 
    thetas: torch.Tensor
    ):
    
    results = {}
    res_support = support_coverage_analysis(posterior_samples, thetas)
    results['support'] = {
        "independent_accuracy": res_support[0],
        "avg_accuracy": res_support[1],
        "joint_accuracy": res_support[2]
    }



    res_1sigma = xsigma_coverage_analysis(posterior_samples, thetas, 0.68)
    results['one_sigma'] = {
        "independent_accuracy": res_1sigma[0],
        "avg_accuracy": res_1sigma[1],
        "joint_accuracy": res_1sigma[2],
        "ci": res_1sigma[3]
    }

    res_2sigma = xsigma_coverage_analysis(posterior_samples, thetas, 0.95)
    results['two_sigma'] = {
        "independent_accuracy": res_2sigma[0],
        "avg_accuracy": res_2sigma[1],
        "joint_accuracy": res_2sigma[2],
        "ci": res_2sigma[3]
    }

    return results