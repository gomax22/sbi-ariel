
import argparse
import os
import numpy as np
import yaml

from dingo.core.posterior_models.build_model import ( # type: ignore
    build_model_from_kwargs,
    autocomplete_model_kwargs,
)
from dingo.core.utils import build_train_and_test_loaders, RuntimeLimits # type: ignore
from dataset import load_dataset
import torch
from tqdm import tqdm
import time

import gc

def compute_posterior_samples(model, settings, dataset, test_loader):
    # dataset is already standardized
    device = settings["training"].get("device", "cpu")
    n_repeats = settings["evaluation"].get("n_repeats", 128) 
    dim_theta = settings["task"].get("dim_theta", 7)

    repeat_step = n_repeats // 16

    posterior_distribution = torch.zeros((len(test_loader.dataset), n_repeats, dim_theta), device=torch.device("cpu"))
    posterior_log_probs = torch.zeros((len(test_loader.dataset), n_repeats), device=torch.device("cpu"))
    # reference_log_probs = torch.zeros((n_repeats, len(test_loader.dataset)), device=device)

    with tqdm(total=n_repeats * len(test_loader.dataset), desc="Computing posterior samples...") as pbar:
        
        for idx, (_, obs) in enumerate(test_loader):
            # batch_size = obs.shape[0]
            
            time_start = time.time()
            obs = obs.repeat((repeat_step, 1)).to(device) # 64
            time_end = time.time()
            print(f"Time taken for repeating batch {idx} n_times={repeat_step}: {time_end - time_start:.2f} s")
            for repeat_idx in range(n_repeats // repeat_step):
                
                time_start = time.time()
                posterior_samples_batch, posterior_log_probs_batch = model.sample_and_log_prob_batch(obs)
                time_end = time.time()
                
                print(f"Time taken for batch {idx} repeat {repeat_idx}: {time_end - time_start:.2f} s")
                posterior_distribution[idx, (repeat_idx*repeat_step):((repeat_idx+1)*repeat_step)] = posterior_samples_batch.detach().cpu()
                posterior_log_probs[idx, (repeat_idx*repeat_step):((repeat_idx+1)*repeat_step)] = posterior_log_probs_batch.detach().cpu()
                pbar.update(repeat_step)

                gc.collect()
                torch.cuda.empty_cache()

    # posterior_samples = torch.cat(posterior_samples, dim=0)

    return posterior_distribution, posterior_log_probs, None


def train_model(train_dir, settings, train_loader, test_loader, use_wandb=False):
    autocomplete_model_kwargs(
        settings["model"],
        input_dim=settings["task"]["dim_theta"],  # input = theta dimension
        context_dim=settings["task"]["dim_x"],  # context dim = observation dimension
    )

    model = build_model_from_kwargs(
        settings={"train_settings": settings},
        device=settings["training"].get("device", "cpu"),
    )

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
        device=settings["training"].get("device", "cpu"),
    )
    return best_model


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train a model")
    ap.add_argument("--run_dir", type=str, required=True, help="Path to the settings file")
    ap.add_argument("--test", action="store_true", help="Compute posterior distribution for test set")
    args = ap.parse_args()


    # load settings
    with open(os.path.join(args.run_dir, "settings.yaml"), "r") as f:
        settings = yaml.safe_load(f)  

    # settings["evaluation"]["n_repeats"] = 2048

    # load dataset
    try:
        test_dataset = load_dataset(settings)
    except FileNotFoundError:
        settings["dataset"]["path"] = "data/"
        test_dataset = load_dataset(settings)

    test_dataset = test_dataset[2] if args.test else test_dataset[1]
    
    # build test loader
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=settings["training"]["num_workers"],
    )

    # load the best model
    best_model = build_model_from_kwargs(
        filename=os.path.join(args.run_dir, "best_model.pt"),
        device=settings["training"].get("device", "cpu"),
    )

    time_start = time.time()
    posterior_distribution, posterior_log_probs, reference_log_probs = compute_posterior_samples(
        best_model, settings, test_dataset, test_loader
    )
    time_end = time.time()

    # posterior_distribution = posterior_distribution.permute(1, 0, 2)
    # posterior_log_probs = posterior_log_probs.permute(1, 0)

    try:
        posterior_distribution = test_dataset.normalize(
            posterior_distribution.cpu(), label="theta", inverse=True
        )
    except KeyError:
        posterior_distribution = test_dataset.standardize(
            posterior_distribution.cpu(), label="theta", inverse=True
        )

    np.save(os.path.join(args.run_dir, "posterior_distribution.npy" if not args.test else "posterior_distribution_test.npy"), posterior_distribution.detach().cpu().numpy()) # n_samples x n_repeats x dim_theta
    np.save(os.path.join(args.run_dir, "posterior_log_probs.npy" if not args.test else "posterior_log_probs_test.npy"), posterior_log_probs.detach().cpu().numpy())
    # np.save(os.path.join(args.run_dir, "reference_log_probs.npy"), reference_log_probs.detach().cpu().numpy())

    # format time in hours, minutes, seconds
    hours, rem = divmod(time_end-time_start, 3600)
    minutes, seconds = divmod(rem, 60)

    time_str = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)
    print("Time taken: {}".format(time_str))

    # update metadata in settings file with time taken to compute posterior
    settings["metadata"]["posterior_computation_time"] = time_str

    # dump settings after training
    with open(os.path.join(args.run_dir, "settings.yaml"), "w") as f:
        yaml.dump(settings, f)