# Taken from: https://github.com/sbi-dev/sbi/blob/main/sbi/diagnostics/sbc.py
# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>


import torch
import numpy as np
# from sbi.analysis.sbc import check_sbc, run_sbc (in 0.23.3 sbi.analysis.sbc -> sbi.diagnostics.sbc)
# from sbi.analysis.sbc import check_sbc, run_sbc

from tqdm import tqdm

import torch
import numpy as np
import warnings
import torch
from scipy.stats import kstest, uniform
from torch import Tensor, ones, zeros
from torch.distributions import Uniform
from sbi.utils.metrics import c2st
from typing import Callable, Tuple, Dict, Union, List

# adapted version of run_sbc 
def run_sbc(
    thetas: torch.Tensor,
    posterior_samples: torch.Tensor,
    reduce_fns: Union[str, Callable, List[Callable]] = "marginals",
) -> Tuple[Tensor, Tensor]:

    num_sbc_samples, num_posterior_samples, dim_theta = posterior_samples.shape

    if num_sbc_samples < 100:
        warnings.warn(
            """Number of SBC samples should be on the order of 100s to give realiable
            results. We recommend using 300."""
        )
    if num_posterior_samples < 100:
        warnings.warn(
            """Number of posterior samples for ranking should be on the order
            of 100s to give reliable SBC results. We recommend using at least 300."""
        )

    if isinstance(reduce_fns, str):
        assert reduce_fns == "marginals", (
            "`reduce_fn` must either be the string `marginals` or a Callable or a List "
            "of Callables."
        )
        reduce_fns = [
            eval(f"lambda theta, x: theta[:, {i}]") for i in range(dim_theta)
        ]


    dap_samples = torch.zeros((num_sbc_samples, dim_theta))
    ranks = torch.zeros((num_sbc_samples, len(reduce_fns)))

    pbar = tqdm(total=num_sbc_samples, desc="Running simulation-based calibration...")
    for idx, (tho, posterior_samples_batch) in enumerate(
        zip(thetas, posterior_samples)):
        
        # Save one random sample for data average posterior (dap).
        sample_idx = np.random.choice(range(num_posterior_samples))
        dap_samples[idx] = posterior_samples_batch[sample_idx]

        # rank for each posterior dimension as in Talts et al. section 4.1.
        for i, reduce_fn in enumerate(reduce_fns):
            ranks[idx, i] = (
                (reduce_fn(posterior_samples_batch, None) < reduce_fn(tho.unsqueeze(0), None)).sum().item()
            )
        pbar.update(1)

    return ranks, dap_samples


def check_sbc(
    ranks: Tensor,
    prior_samples: Tensor,
    dap_samples: Tensor,
    num_posterior_samples: int = 1000,
    num_c2st_repetitions: int = 1,
) -> Dict[str, Tensor]:
    """Return uniformity checks and data averaged posterior checks for SBC.

    Args:
        ranks: ranks for each sbc run and for each model parameter, i.e.,
            shape (N, dim_parameters)
        prior_samples: N samples from the prior
        dap_samples: N samples from the data averaged posterior
        num_posterior_samples: number of posterior samples used for sbc ranking.
        num_c2st_repetitions: number of times c2st is repeated to estimate robustness.

    Returns (all in a dictionary):
        ks_pvals: p-values of the Kolmogorov-Smirnov test of uniformity,
            one for each dim_parameters.
        c2st_ranks: C2ST accuracy of between ranks and uniform baseline,
            one for each dim_parameters.
        c2st_dap: C2ST accuracy between prior and dap samples, single value.
    """
    if ranks.shape[0] < 100:
        warnings.warn(
            """You are computing SBC checks with less than 100 samples. These checks
            should be based on a large number of test samples theta_o, x_o. We
            recommend using at least 100."""
        )

    ks_pvals = check_uniformity_frequentist(ranks, num_posterior_samples)
    c2st_ranks = check_uniformity_c2st(
        ranks, num_posterior_samples, num_repetitions=num_c2st_repetitions
    )
    c2st_scores_dap = check_prior_vs_dap(prior_samples, dap_samples)

    return dict(
        ks_pvals=ks_pvals,
        c2st_ranks=c2st_ranks,
        c2st_dap=c2st_scores_dap,
    )


def check_prior_vs_dap(prior_samples: Tensor, dap_samples: Tensor) -> Tensor:
    """Returns the c2st accuracy between prior and data avaraged posterior samples.

    c2st is calculated for each dimension separately.

    According to simulation-based calibration, the inference methods is well-calibrated
    if the data averaged posterior samples follow the same distribution as the prior,
    i.e., if the c2st score is close to 0.5. If it is not, then this suggests that the
    inference method is not well-calibrated (see Talts et al, "Simulation-based
    calibration" for details).
    """

    assert prior_samples.shape == dap_samples.shape

    return torch.tensor(
        [
            c2st(s1.unsqueeze(1), s2.unsqueeze(1))
            for s1, s2 in zip(prior_samples.T, dap_samples.T)
        ]
    )


def check_uniformity_frequentist(ranks, num_posterior_samples) -> Tensor:
    """Return p-values for uniformity of the ranks.

    Calculates Kolomogorov-Smirnov test using scipy.

    Args:
        ranks: ranks for each sbc run and for each model parameter, i.e.,
            shape (N, dim_parameters)
        num_posterior_samples: number of posterior samples used for sbc ranking.

    Returns:
        ks_pvals: p-values of the Kolmogorov-Smirnov test of uniformity,
            one for each dim_parameters.
    """
    kstest_pvals = torch.tensor(
        [
            kstest(rks, uniform(loc=0, scale=num_posterior_samples).cdf)[1]
            for rks in ranks.T
        ],
        dtype=torch.float32,
    )

    return kstest_pvals


def check_uniformity_c2st(
    ranks, num_posterior_samples, num_repetitions: int = 1
) -> Tensor:
    """Return c2st scores for uniformity of the ranks.

    Run a c2st between ranks and uniform samples.

    Args:
        ranks: ranks for each sbc run and for each model parameter, i.e.,
            shape (N, dim_parameters)
        num_posterior_samples: number of posterior samples used for sbc ranking.
        num_repetitions: repetitions of C2ST tests estimate classifier variance.

    Returns:
        c2st_ranks: C2ST accuracy of between ranks and uniform baseline,
            one for each dim_parameters.
    """

    c2st_scores = torch.tensor(
        [
            [
                c2st(
                    rks.unsqueeze(1),
                    Uniform(zeros(1), num_posterior_samples * ones(1)).sample(
                        torch.Size((ranks.shape[0],))
                    ),
                )
                for rks in ranks.T
            ]
            for _ in range(num_repetitions)
        ]
    )

    # Use variance over repetitions to estimate robustness of c2st.
    if (c2st_scores.std(0) > 0.05).any():
        warnings.warn(
            f"""C2ST score variability is larger than {0.05}: std={c2st_scores.std(0)},
            result may be unreliable. Consider increasing the number of samples.
            """
        )

    # Return the mean over repetitions as c2st score estimate.
    return c2st_scores.mean(0)

