
import argparse
import math
import os
import torch
import numpy as np
import yaml
from tqdm import tqdm
from dataset import ArielDataset, load_dataset
import time
from datetime import timedelta
from dingo.core.posterior_models.build_model import build_model_from_kwargs # type: ignore

import torch.distributed as dist
import torch.multiprocessing as mp
import gc
import random


# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
random.seed(SEED)

def compute_posterior_distribution(
    model,
    run_dir: str, 
    test: bool,
    settings: dict, 
    dataset: ArielDataset, 
    test_loader: torch.utils.data.DataLoader,
    rank: int,
    world_size: int
):
    
    n_repeats = settings["evaluation"].get("n_repeats", 2048) 
    dim_theta = settings["task"].get("dim_theta", 7)
    # device = settings["training"].get("device", "cpu")
    device = torch.device(f"cuda:{rank}")
    
    # for each datapoint in the test_loader
        # load number of likelihood evaluations from nested sampling tracedata
        # sample from the model same amount of times

    # tracedata
    repeat_step = n_repeats // world_size # 2048 / 4 = 512

    posterior_distribution = torch.zeros((len(test_loader.dataset), repeat_step, dim_theta), device=device)     
    posterior_log_probs = torch.zeros((len(test_loader.dataset), repeat_step), device=device) 
    # reference_log_probs = torch.zeros((repeat_step, len(test_loader.dataset)), device=device)
    
    pbar = tqdm(total=repeat_step * len(test_loader.dataset), desc="Computing posterior samples...", position=rank, leave=True)

    for idx, (_, obs) in enumerate(test_loader):
        batch_size = repeat_step // 8  # 512 / 8 = 64

        time_start = time.time()
        obs = obs.repeat((batch_size, 1)).to(device) # 64
        time_end = time.time()
        print(f"Time taken for repeating batch {idx} n_times={batch_size}: {time_end - time_start:.2f} s")
        
        for repeat_idx in range(repeat_step // batch_size):
            
            time_start = time.time()
            posterior_samples_batch, posterior_log_probs_batch = model.sample_and_log_prob_batch(obs)
            time_end = time.time()
            
            print(f"Time taken for batch {idx} repeat {repeat_idx} /{repeat_step // batch_size}: {time_end - time_start:.2f} s")
            posterior_distribution[idx, (repeat_idx*batch_size):((repeat_idx+1)*batch_size)] = posterior_samples_batch.detach()
            posterior_log_probs[idx, (repeat_idx*batch_size):((repeat_idx+1)*batch_size)] = posterior_log_probs_batch.detach()
            pbar.update(batch_size)

            gc.collect()
            torch.cuda.empty_cache()
        # torch.cuda.synchronize(device=rank) # dist.barrier() not working
    
    # posterior_distribution = posterior_distribution.permute(1, 0, 2)
    # posterior_log_probs = posterior_log_probs.permute(1, 0)
        
    
    # gather and save posterior distribution
    if rank == 0:
        gathered_posterior_distribution = [torch.zeros_like(posterior_distribution, device=device) for _ in range(world_size)]
        gathered_posterior_log_probs = [torch.zeros_like(posterior_log_probs, device=device) for _ in range(world_size)]
        # gathered_reference_log_probs = [torch.zeros_like(reference_log_probs, device=device) for _ in range(world_size)]
    else:
        gathered_posterior_distribution = None
        gathered_posterior_log_probs = None
        # gathered_reference_log_probs = None
    
    # torch.cuda.synchronize(device=rank)
    dist.gather(posterior_distribution, gathered_posterior_distribution, dst=0)
    if rank == 0:
        posterior_distribution = torch.cat(gathered_posterior_distribution, dim=0)
        
         
        try:
            posterior_distribution = dataset.normalize(
                posterior_distribution.cpu(), "theta", inverse=True
            )
        except KeyError:
            posterior_distribution = dataset.standardize(
                posterior_distribution.cpu(), "theta", inverse=True
            )

        np.save(os.path.join(run_dir, 'posterior_distribution.npy' if not test else 'posterior_distribution_test.npy'), posterior_distribution.detach().cpu().numpy()) # n_samples, n_repeats, dim_theta
        print(f"Rank {rank} - posterior_distribution shape: {posterior_distribution.shape}")

    # gather and save posterior log probs
    # torch.cuda.synchronize(device=rank)
    dist.gather(posterior_log_probs, gathered_posterior_log_probs, dst=0)
    if rank == 0:
        posterior_log_probs = torch.cat(gathered_posterior_log_probs, dim=0)
        np.save(os.path.join(run_dir, 'posterior_log_probs.npy' if not test else 'posterior_log_probs_test.npy'), posterior_log_probs.detach().cpu().numpy())
        print(f"Rank {rank} - posterior_log_probs shape: {posterior_log_probs.shape}")


def setup(rank, world_size, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port
    # os.environ["TORCH_CPP_LOG_LEVEL"] = "INFO"
    # os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size, timeout=timedelta(hours=4))

def cleanup():
    dist.destroy_process_group()



def distributed_compute_posterior_distribution(rank, world_size, settings, run_dir, test, latest, port):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size, port)
    
    test_dataset = load_dataset(settings)
    test_dataset = test_dataset[2] if test else test_dataset[1]

    print(f"Rank {rank} - dataset length: {len(test_dataset)}")
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=1, 
        shuffle=False,
        num_workers=settings["training"]["num_workers"],
    )

    model = build_model_from_kwargs(
        filename=os.path.join(run_dir, "best_model.pt" if not latest else "model_latest.pt"),
        device=settings["training"].get("device", "cpu"),
    )

    # create model and move it to GPU with id rank
    # model.network = DDP(model.network, device_ids=[rank])

    model.device = torch.device(f"cuda:{rank}")
    model.network.continuous_flow = model.network.continuous_flow.to(model.device)
    model.network.theta_embedding_net = model.network.theta_embedding_net.to(model.device)
    model.network.context_embedding_net = model.network.context_embedding_net.to(model.device)

    gc.collect()
    torch.cuda.empty_cache()

    compute_posterior_distribution(model, run_dir, test, settings, test_dataset, test_loader, rank, world_size)
    
    cleanup()
    print(f"Finished running basic DDP example on rank {rank}.")


def run_distributed_demo(demo_fn, world_size, settings, run_dir, latest, port):
    mp.spawn(demo_fn,
             args=(world_size, settings, run_dir, latest, port),
             nprocs=world_size,
             join=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True, help="Base save directory for the evaluation")
    parser.add_argument("--test", action="store_true", help="Compute posterior distribution for ADC test set")
    parser.add_argument("--latest", action="store_true", default=False, help="Use the latest model in the directory")
    parser.add_argument("--port", type=str, default="12355", help="Port for DDP")
    args = parser.parse_args()

    with open(os.path.join(args.run_dir, "settings.yaml"), "r") as f:
        settings = yaml.safe_load(f)

    settings["evaluation"]["n_repeats"] = 2048
    settings["training"]["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    world_size = torch.cuda.device_count()

    time_start = time.time()
    run_distributed_demo(distributed_compute_posterior_distribution, world_size, settings, **vars(args))
    # compute_posterior_distribution(ddp_model, args.run_dir, settings, valid_dataset, valid_loader)
    time_end = time.time()
    
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

