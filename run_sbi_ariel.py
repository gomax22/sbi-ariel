
import argparse
from datetime import datetime
import os
from pathlib import Path
import numpy as np
import torch

import yaml
from dingo.core.posterior_models.build_model import ( # type: ignore
    build_model_from_kwargs,
    autocomplete_model_kwargs,
)
from dingo.core.utils import build_train_and_test_loaders, RuntimeLimits # type: ignore
from dataset import load_dataset
import random

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
random.seed(SEED)

def train_model(train_dir, settings, train_loader, test_loader, use_wandb=False):
    autocomplete_model_kwargs(
        settings["model"],
        input_dim=settings["task"]["dim_theta"],  # input = theta dimension
        context_dim=settings["task"]["dim_x"],  # context dim = observation dimension
    )

    model = build_model_from_kwargs(
        settings={"train_settings": settings},
        device="cpu",
    )

    model.device = torch.device(settings["training"].get("device", "cpu"))

    model.network.continuous_flow = model.network.continuous_flow.to(model.device)
    model.network.theta_embedding_net = model.network.theta_embedding_net.to(model.device)
    model.network.context_embedding_net = model.network.context_embedding_net.to(model.device)

    # Before training you need to call the following lines:
    model.optimizer_kwargs = settings["training"]["optimizer"]
    model.scheduler_kwargs = settings["training"]["scheduler"]
    model.initialize_optimizer_and_scheduler()

    # train model
    runtime_limits = RuntimeLimits(
        epoch_start=0,
        max_epochs_total=settings["training"]["epochs"],
    )
    model.train(
        train_loader,
        test_loader,
        train_dir=train_dir,
        runtime_limits=runtime_limits,
        early_stopping=settings["training"]["early_stopping"],
        patience=settings["training"]["patience"],
        use_wandb=use_wandb,
    )

    # load the best model
    best_model = build_model_from_kwargs(
        filename=os.path.join(train_dir, "best_model.pt"),
        device="cpu",
    )

    best_model.device = torch.device(settings["training"].get("device", "cpu"))

    best_model.network.continuous_flow = best_model.network.continuous_flow.to(best_model.device)
    best_model.network.theta_embedding_net = best_model.network.theta_embedding_net.to(best_model.device)
    best_model.network.context_embedding_net = best_model.network.context_embedding_net.to(best_model.device)


    return best_model


def run_sbi_ariel(settings_file, experiments_dir):
    
    current_time = datetime.now()
    working_dir, fname = os.path.split(os.path.abspath(__file__))

    
    # load settings
    with open(settings_file, "r") as f:
        settings = yaml.safe_load(f)  

    # load dataset
    train_dataset, test_dataset, _ = load_dataset(settings)
    
    # build train and valid loaders
    train_loader, valid_loader = build_train_and_test_loaders(
        train_dataset,
        settings["training"]["train_fraction"],
        settings["training"]["batch_size"],
        settings["training"]["num_workers"],
    )
    

    # runs/
    #       p_t/
    #           ctxt_u
    #               theta_emb_v/
    #                   layers_x/
    #                       a_y/
    #                           lr_z/
    

    # context = "full" if "full" in settings["dataset"]["type"].lower() else "aux_data"
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

    rundir_name = os.path.join(f"{experiments_dir}",
                                f"dt_{dataset_type}",
                                f"p_{prior}",
                                # f"ctxt_{context}", 
                                f"m_{model_architecture}",
                                f"theta_emb_{theta_emb}",
                                f"layers_{num_layers}", 
                                f"mfl_{multiplicative_factor_layers}",
                                f"dropout_{dropout}",
                                f"a_{alpha}", 
                                f"lr_{learning_rate}") 
    output_dir = os.path.join(working_dir, rundir_name, current_time.strftime('%Y%m%d_%H%M%S'))
    
    # create directory for training    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
        
    # adjust hidden_dims for the model
    hidden_dims_layers = settings["model"]["posterior_kwargs"]["hidden_dims_layers"]
    multiplicative_factor_layers = settings["model"]["posterior_kwargs"]["multiplicative_factor_layers"]
    hidden_dims = [[hidden_dims_layers[idx]] * multiplicative_factor_layers for idx in range(len(settings["model"]["posterior_kwargs"]["hidden_dims_layers"]))]
    settings["model"]["posterior_kwargs"]["hidden_dims"] = [int(x) for x in np.array(hidden_dims).flatten()]

    # copy settings to output directory before training
    with open(os.path.join(output_dir, "settings.yaml"), "w") as f:
        yaml.dump(settings, f)

    
    # train model
    model = train_model(
        train_dir=output_dir,
        settings=settings,
        train_loader=train_loader,
        test_loader=valid_loader,
        )

    # load the best model
    best_model = torch.load(os.path.join(output_dir, "best_model.pt"), map_location=settings["training"].get("device", "cpu"))

    # update settings with metadata
    settings["metadata"] = {
        "settings_file": os.path.abspath(settings_file),
        "file": os.path.abspath(__file__),
        "best_model_epoch": best_model["epoch"],
    }

    # reset device
    settings["training"]["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    # dump settings after training
    with open(os.path.join(output_dir, "settings.yaml"), "w") as f:
        yaml.dump(settings, f)



# 480 configurations to explore... we need organization.
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train a model")
    ap.add_argument("--settings_file", type=str, required=True, help="Path to the settings file")
    ap.add_argument("--experiments_dir", type=str, required=False, default='runs', help="Path to the directory where experiments will be stored")
    args = vars(ap.parse_args())
    run_sbi_ariel(**args)
