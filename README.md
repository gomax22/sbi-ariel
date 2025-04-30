# Flow Matching Posterior Estimation for Simulation-based Atmospheric Retrieval of Exoplanets
Official code repository of the paper "Flow Matching Posterior Estimation for Simulation-based Atmospheric Retrieval of Exoplanets", submitted to IEEE Access (Applied Research)


## Abstract
The characterization of exoplanetary atmospheres allows a deeper understanding of planetary formation, evolution, and habitability through _atmospheric retrieval_, which consists in inferring various properties of exoplanetary atmospheres given their spectroscopic observations.
Traditional atmospheric retrieval methods based on Bayesian inference, such as Nested Sampling, require significant computational resources to compute the full posterior distribution of atmospheric parameters, limiting their scalability for future large-scale surveys and high-resolution characterizations. 
Additionally, the rise of modern density estimation techniques poses a fundamental need for comprehensive evaluation frameworks to objectively compare the posterior distributions of heterogeneous probabilistic estimators.
Within the scope of the 2023 edition of the Ariel Data Challenge, this work proposes a novel, scalable atmospheric retrieval framework based on Flow Matching Posterior Estimation (FMPE) and Continuous Normalizing Flows (CNFs), leveraging transmission spectra, instrumental uncertainties across wavelength channels, and auxiliary information about planetary systems, to retrieve the posterior distribution of atmospheric parameters in a significantly reduced computational time compared to conventional techniques.
Through the fair definition of an extensive posterior evaluation framework, our approach demonstrates superior performance in terms of target prediction error, uncertainty quantification, calibration, and posterior coverage, outperforming existing neural-based and sampling-based retrieval methods. 
In addition, the integration of auxiliary planetary system information into the proposed retrieval framework enhances predictive accuracy, reduces uncertainty, and improves interpretability, bridging data-driven models with physical mechanisms.

## Installation

```
conda env create -f environment.yml
conda activate fmpe-ariel
```


## Workflow
This section aims to provide guidelines for training, testing, and evaluating models and to reproduce the experiments presented in the paper.

### Data

The ADC2023 dataset is publicly available at this link.

To preprocess the downloaded dataset, run the script ``make_dataset.py`` (see ``--help`` for additional information)-

Datasets are available on request (both in the original version and/or the preprocessed one.)

### Training
To train the CNFs with FMPE, we offer two options:
- **Single-configuration training:** the script ``run_sbi_ariel.py`` performs the training of a single model configuration by specifying
    *  a settings file with the option ``--settings_file``(typically placed in ``settings/training/config.yaml``)
    * an output directory for storing the model checkpoints and related information.
- **Multiple-configuration training:** the script ``run_sbi_ariel_batch.py`` performs the traning of multiple model configurations by specifying 
    * the sweep values for the hyperparameters in the script file
    * a settings dir with the option ``--settings_dir`` (default: ``settings/training/base_settings.yaml``), where temporary settings files will be stored.
    * the root directory (with the option ``--source_dir``) where experiments will be stored.

### Filtering the runs in terms of validation loss (optional)

To find the best runs in terms of validation loss, run the script ``find_best_runs.py`` by specifying:
* the directory where the multiple training runs are stored with the option ``--runs_dir``
* the number of best configurations to return for each dataset with the option ``--top-k`` (you can comment out the unnecessary scans).

For example, the command:

```bash
python find_best_runs --runs_dir <your_source_dir> --top-k 1
```

provides the best runs in terms of validation loss within the given directory.


### Posterior computation

To infer the posterior distribution of atmospheric parameters using the pre-trained models (or other estimators) on the samples of the designed test set, we provide multiple options:
- **Single-GPU posterior computation** with the script ``compute_posterior_distribution_single_gpu.py``
- **Multiple-GPU posterior computation** with the script ``compute_posterior_distribution.py``

Both scripts can be executed by specifying:
* the directory where a given model (typically saved with the filename ``best_model.pt``) is stored with the option ``--run_dir``.
* (optional) the option ``--test`` performs the posterior computation on the samples of the effective test set of the ADC2023 (even though, we lack of the Nested Sampling-based posterior.)

N.B.: it is crucial that the posterior samples are saved as ``posterior_distribution.npy`` (of shape $N_{test}$, R, d$) where $N_{test} is the number of test samples, $R$ is the number of realizations (fixed to 2048 in our case), and $d$ is the number of target atmospheric parameters (i.e., 7 in our case).

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

## Citation

```bibtex
T.B.D
```
