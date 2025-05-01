# Flow Matching Posterior Estimation for Simulation-based Atmospheric Retrieval of Exoplanets

![Python 3.10](https://img.shields.io/badge/python-3.10+-blue)

This repository contains the code for the research paper:

> M. Giordano Orsini, A. Ferone, L. Inno, A. Casolaro, A. Maratea (2025).
> "Flow Matching Posterior Estimation for Simulation-based Atmospheric Retrieval of Exoplanets".
> Submitted to IEEE Access (Applied Research)


---

## Installation
Our code was developed on a machine with the following hardware specifications:
- **CPU**: 13th Gen Intel Core i9-13900KF (24 cores @ 5.8 GHz)
- **GPU**: NVIDIA GeForce RTX 4090 (24GB VRAM)
- **OS**: Ubuntu 22.04 LTS

and makes use of the following software libraries among the others:
  - `dingo` ([Dax et al. 2021](https://arxiv.org/abs/2106.12594))
  - `PyTorch` (v2.0+ with CUDA 12.4)
  - `netcal` ([Küppers et al. 2022](http://arxiv.org/abs/2207.01242)) for calibration metrics
  - `sbi` ([Tejero-Cantero et al. 2020](https://joss.theoj.org/papers/10.21105/joss.02505)) for statistical inference

To create a virtual environment for running the code, please execute the following commands:

```bash
git clone https://github.com/gomax22/sbi-ariel
cd sbi-ariel
conda env create -f environment.yml
conda activate fmpe-ariel
```


## Workflow
This section aims to provide guidelines for training, testing, and evaluating models and to reproduce the experiments presented in the paper.

### Data

The ADC2023 dataset is publicly available at this [link](https://www.ariel-datachallenge.space/ML/download/).

To preprocess the downloaded dataset, run the script ``make_dataset.py`` (see ``--help`` for additional information).

If needed, original and/or preprocessed datasets are available on request.

### Settings file
Several stages of the proposed retrieval framework are guided by specifying an initial settings file, organized as follows:
- dataset
- evaluation
- model
- task
- training


Please, see the file ``settings/training/custom.yaml`` for additional information.

### Training
To train the CNFs with FMPE, we offer two options:
- **Single-configuration training:** the script ``run_sbi_ariel.py`` performs the training of a single model configuration by specifying
    *  a settings file with the option ``--settings_file``(typically placed in ``settings/training/base_settings.yaml``)
    * an output directory for storing the experimental outcomes (e..g, model checkpoints, etc.) with the option ``--experiments_dir``.

    For example,
    ```bash
    python run_sbi_ariel.py --settings_file settings/training/base_settings.yaml --experiments_dir runs
    ```    

- **Multiple-configuration training:** the script ``run_sbi_ariel_batch.py`` performs the traning of multiple model configurations by specifying 
    * the sweep values for the hyperparameters in the script file
    * a settings dir with the option ``--settings_dir`` (default: ``settings/training/base_settings.yaml``), where temporary settings files will be stored.
    * the root directory (with the option ``--experiments_dir``) where experiments will be stored.

    For example,
    ```bash
    python run_sbi_ariel_batch.py --settings_dir settings/training --experiments_dir batch_runs
    ```    
    **N.B.:** Be careful as this could be decidedly time-intensive.

### Filtering the runs in terms of validation loss (optional)

To find the best runs in terms of validation loss, run the script ``find_best_runs.py`` by specifying:
* the directory where the multiple training runs are stored with the option ``--runs_dir``
* the number of best configurations to return for each dataset with the option ``--top-k`` (you can comment out the unnecessary scans).

For example, the command:

```bash
python find_best_runs --runs_dir batch_runs --top-k 1
```

provides the best runs in terms of validation loss within the given directory.


### Posterior computation

To infer the posterior distribution of atmospheric parameters using the pre-trained models (or other estimators) on the samples of the designed test set, we provide multiple options:
- **Single-GPU posterior computation** with the script ``compute_posterior_distribution_single_gpu.py``
- **Multiple-GPU posterior computation** with the script ``compute_posterior_distribution.py``

Both scripts can be executed by specifying:
* the directory where a given model (typically saved with the filename ``best_model.pt``) is stored with the option ``--run_dir``.
* (optional) the option ``--test`` performs the posterior computation on the samples of the effective test set of the ADC2023 (even though, we lack of the Nested Sampling-based posterior.)

For example, the command:
 ```bash
python compute_posterior_distribution_single_gpu.py --run_dir path/to/training/run 
```
produces the posterior distribution (saved as ``posterior_distribution.npy`` within the corresponding directory) of shape $N_{test}$, R, d$ where:
* $N_{test}$ is the number of test samples,
* $R$ is the number of realizations (fixed to 2048 in the original settings file), and
* $d$ is the number of target atmospheric parameters (i.e., 7 in the ADC2023 case).

**WARNING:** be careful as this could take several hours.
### Evaluation

To evaluate the predicted posterior distribution, we offer several options:
- **Scores of the ADC2023**
    * first, extract the posterior distribution computed by Nested Sampling with the script ``compute_reference_distribution.py``
    * the script ``adc_evaluation.py`` evaluates the predicted posterior distribution (given the samples of the designed test set) according to the ADC2023 scoring system.
- **Prediction error**
    * the script ``evaluate_losses.py`` evaluates the predicted posterior distribution in terms of prediction errors on target parameters and spectra, according to common metrics.
- **Calibration**
    * the script ``calibration.py`` evaluates the calibration performance of the predicted posterior distribution according to popular metrics in this context.
    * the script ``evaluate_sbc_tarp.py`` runs simulation-based calibration and TARP on the predicted posterior distribution.
- **Posterior Coverage Analysis (or ground-truth benchmarking)**
    * the script ``coverage_evaluation.py`` evaluates the coverage performance of the predicted posterio distribution under several coverage levels.

Run the scripts with the option ``--help`` for additional information.

### Comparison (optional)
Once the predicted posterior distribution
To perform a comparison between multiple trained models and other estimators according to the extended posterior evaluation framework, run the script ``comparison.py`` by specifying:
* a settings file for aggregating information about the competing methods (e.g., see ``settings/comparison/final_comparison_settings.yaml``).
* the directory where results are stored with the option ``--output_dir``.

For example,
```bash
python comparison.py --settings_file settings/comparison/final_comparison_settings.yaml --output_dir comparisons/final
```

## Citation

```bibtex
T.B.D
```
