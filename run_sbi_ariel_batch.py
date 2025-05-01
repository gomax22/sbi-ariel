
import argparse
from datetime import datetime
import os
from pathlib import Path
from matplotlib.dates import DAILY
import numpy as np
import torch
import random
import yaml
from tqdm import tqdm
from run_sbi_ariel import run_sbi_ariel
import concurrent
import concurrent.futures
import gc

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
random.seed(SEED)


PRIOR = "StandardNormal"
DROPOUT = [0.0, 0.1]
TIME_PRIOR_EXPONENT = [-0.75, -0.5, -0.25, 0.0, 0.5, 1.0, 2.0, 4.0]
LEARNING_RATE = [0.001, 0.0005, 0.0001]
HIDDEN_DIMS_LAYERS = [[128, 256, 512, 1024, 512, 256, 128], [512] * 7]
MULTIPLICATIVE_FACTOR_LAYERS = [1, 2]
DATASET_TYPE = [
    "FullNormalizedArielDataset", 
    "RealFullNormalizedArielDataset", 
    "RealNormalizedArielDataset",
    "FullStandardizedArielDataset",
    "RealFullStandardizedArielDataset",
    "RealStandardizedArielDataset"
    "NormalizedArielDataset",
    "StandardizedArielDataset",
    "NoisyNormalizedArielDataset",
    "NoisyStandardizedArielDataset",
    "RealNoisyNormalizedArielDataset",
    "RealNoisyStandardizedArielDataset",
]


def build_settings_fname(source_dir, settings):
    dataset_type = settings["dataset"]["type"]
    learning_rate = settings["training"]["optimizer"]["lr"]
    alpha = settings["model"]["posterior_kwargs"]["time_prior_exponent"]
    dropout = settings["model"]["posterior_kwargs"]["dropout"]
    multiplicative_factor_layers = settings["model"]["posterior_kwargs"]["multiplicative_factor_layers"]
    theta_embedding_hidden_dims = len(settings["model"]["posterior_kwargs"]["theta_embedding_kwargs"]["embedding_net"]["hidden_dims"]) \
        if "theta_embedding_kwargs" in settings["model"]["posterior_kwargs"] \
        else 0
    
    # posterior_model_hidden_dims * multiplicative_factor_layers + theta_embedding_hidden_dims
    num_layers = len(settings["model"]["posterior_kwargs"]["hidden_dims_layers"]) * \
        settings["model"]["posterior_kwargs"]["multiplicative_factor_layers"] + \
        theta_embedding_hidden_dims
    
    theta_emb = 1 if "theta_embedding_kwargs" in settings["model"]["posterior_kwargs"] else 0 
    prior = settings["model"]["prior"]["type"]

    model_architecture = "plain" if len(set(settings["model"]["posterior_kwargs"]["hidden_dims_layers"])) == 1 else "autoencoder"
    fname = f"dt_{dataset_type}_p_{prior}_m_{model_architecture}_theta_emb_{theta_emb}_mfl_{multiplicative_factor_layers}_layers_{num_layers}_dropout_{dropout}_a_{alpha}_lr_{learning_rate}.yaml"
    fname = os.path.join(source_dir, fname)
    return fname


def create_settings(basic_settings, settings_dir):

    temp_dir = os.path.join(settings_dir, "temp_noisy")

    # if Path(temp_dir).exists():
    #     print("Settings directory already exists. Skipping settings creation.")
    #     return [os.path.join(temp_dir, f) for f in os.listdir(temp_dir)]
    
    Path(temp_dir).mkdir(parents=True, exist_ok=True)

    total_it = len(DATASET_TYPE) * len(DROPOUT) * len(TIME_PRIOR_EXPONENT) * len(LEARNING_RATE) * len(HIDDEN_DIMS_LAYERS) * len(MULTIPLICATIVE_FACTOR_LAYERS)
    pbar = tqdm(total=total_it, desc="Building settings files...")
    iters = 0
    settings_files = []
    for dataset_type in DATASET_TYPE:
        for dropout in DROPOUT:
            for alpha in TIME_PRIOR_EXPONENT:
                for learning_rate in LEARNING_RATE:
                    for hidden_dims_layers in HIDDEN_DIMS_LAYERS:
                        for multiplicative_factor_layers in MULTIPLICATIVE_FACTOR_LAYERS:
                            settings = basic_settings.copy()
                            settings["model"]["prior"]["type"] = PRIOR
                            settings["dataset"]["type"] = dataset_type
                            settings["model"]["posterior_kwargs"]["dropout"] = dropout
                            settings["model"]["posterior_kwargs"]["time_prior_exponent"] = alpha
                            settings["training"]["optimizer"]["lr"] = learning_rate
                            settings["model"]["posterior_kwargs"]["hidden_dims_layers"] = hidden_dims_layers
                            settings["model"]["posterior_kwargs"]["multiplicative_factor_layers"] = multiplicative_factor_layers
                            # settings["training"]["device"] = f"cuda:{iters % 4}"
                            settings["training"]["device"] = "cuda:0"
                            settings_fname = build_settings_fname(temp_dir, settings)
                            settings_files.append(settings_fname)

                            with open(settings_fname, "w") as f:
                                yaml.dump(settings, f)
                            
                            pbar.update(1)
                            iters += 1

    pbar.close()
    return settings_files


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train a model")
    ap.add_argument("--settings_dir", type=str, required=True, help="Path to the settings directory")
    ap.add_argument("--experiments_dir", type=str, required=False, default='batch_runs', help="Path to the directory where the experiments will be stored")
    args = ap.parse_args()

    current_time = datetime.now()
    working_dir, fname = os.path.split(os.path.abspath(__file__))

    with open(os.path.join(args.settings_dir, "base_settings.yaml"), "r") as f:
        base_settings = yaml.safe_load(f)

    settings_files = create_settings(base_settings, args.settings_dir)
    
    # settings_files = [[settings_files[0::4]], [settings_files[1::4]], [settings_files[2::4]], [settings_files[3::4]]]
    # settings_files = settings_files[args.device::4]
    
    pbar = tqdm(total=len(settings_files), desc="Running sbi-ariel...")
    with concurrent.futures.ProcessPoolExecutor(max_workers=12) as executor:
        future_to_settings = {executor.submit(run_sbi_ariel, settings_file, args.experiments_dir): settings_file for settings_file in settings_files}
        for future in concurrent.futures.as_completed(future_to_settings):

            # flush cpu and gpu ram
            torch.cuda.empty_cache()
            gc.collect()

            settings_file = future_to_settings[future]
            try:
                data = future.result()
            except Exception as exc:
                print('%r generated an exception: %s' % (settings_file, exc))
            pbar.update(1)
    pbar.close()

    