import argparse
import yaml
import os
import pandas as pd
from tqdm import tqdm
import numpy as np

def get_best_runs(run_dir):

    splits = run_dir.split(os.sep)
    if not any([s.startswith("dt_") for s in splits]):
        raise ValueError("The run directory must contain the dataset name")
    dataset = [s for s in splits if s.startswith("dt_")][0]
    
    best_runs = []
    total = sum([1 for r, d, files in os.walk(run_dir) for f in files if f == "history.txt"])
    with tqdm(total=total, desc="Scanning runs...") as pbar:

        # scan all subdirectories
        for root, dirs, files in os.walk(run_dir):
            for file in files:
                if file != "history.txt":
                    continue

                # load history
                with open(os.path.join(root, "history.txt"), "r") as f:
                    history = pd.read_csv(f, sep="\t", header=None)
                    history = history.dropna().to_numpy()
                    num_epochs = history.shape[0]

                # load best model epoch
                with open(os.path.join(root, "settings.yaml"), "r") as f:
                    settings = yaml.safe_load(f)
                    best_model_epoch = settings["metadata"]["best_model_epoch"]
                    
                if best_model_epoch > num_epochs:
                    pbar.update(1)
                    continue

                best_runs.append((dataset,  root, best_model_epoch, history[best_model_epoch-1,2]))
                pbar.update(1)
    
    best_runs = sorted(best_runs, key=lambda x: x[3])
    return best_runs



def show_best_runs(best_runs, top_k):
    print("Best runs:")
    for br in best_runs:
        for dt, run, epoch, loss in br[:top_k]:
            print(f"Dataset: {dt}, Run: {f'{os.sep}'.join(run.split(os.sep))}, Epoch: {epoch}, Loss: {loss}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Looking for the best runs")
    ap.add_argument("--runs_dir", type=str, required=True, help="Path to the runs directory")
    ap.add_argument("--top-k", type=int, default=5, help="Number of best runs to show")
    args = ap.parse_args()

    best_runs = []
    best_runs.append(
        get_best_runs(
            os.path.join(
                args.runs_dir, 
                "dt_FullNormalizedArielDataset", 
                "p_StandardNormal", 
                "m_plain")
            )[:args.top_k]
        )

    
    best_runs.append(
        get_best_runs(
            os.path.join(
                args.runs_dir, 
                "dt_FullNormalizedArielDataset", 
                "p_StandardNormal", 
                "m_autoencoder")
            )[:args.top_k]
        )
    
    best_runs.append(
        get_best_runs(
            os.path.join(
                args.runs_dir, 
                "dt_FullStandardizedArielDataset", 
                "p_StandardNormal", 
                "m_plain")
            )[:args.top_k]
        )
    
    best_runs.append(
        get_best_runs(
            os.path.join(
                args.runs_dir, 
                "dt_FullStandardizedArielDataset", 
                "p_StandardNormal", 
                "m_autoencoder")
            )[:args.top_k]
        )
    
    best_runs.append(
        get_best_runs(
            os.path.join(
                args.runs_dir, 
                "dt_StandardizedArielDataset", 
                "p_StandardNormal", 
                "m_plain")
            )[:args.top_k]
        )
    
    best_runs.append(
        get_best_runs(
            os.path.join(
                args.runs_dir, 
                "dt_StandardizedArielDataset", 
                "p_StandardNormal", 
                "m_autoencoder")
            )[:args.top_k]
        )
    
    best_runs.append(
        get_best_runs(
            os.path.join(
                args.runs_dir, 
                "dt_RealFullNormalizedArielDataset", 
                "p_StandardNormal", 
                "m_plain")
            )[:args.top_k]
        )
    
    best_runs.append(
        get_best_runs(
            os.path.join(
                args.runs_dir, 
                "dt_RealFullNormalizedArielDataset", 
                "p_StandardNormal", 
                "m_autoencoder")
            )[:args.top_k]
        )
    best_runs.append(
        get_best_runs(
            os.path.join(
                args.runs_dir, 
                "dt_RealFullStandardizedArielDataset", 
                "p_StandardNormal", 
                "m_plain")
            )[:args.top_k]
        )
    
    best_runs.append(
        get_best_runs(
            os.path.join(
                args.runs_dir, 
                "dt_RealFullStandardizedArielDataset", 
                "p_StandardNormal", 
                "m_autoencoder")
            )[:args.top_k]
        )
    
    best_runs.append(
        get_best_runs(
            os.path.join(
                args.runs_dir, 
                "dt_RealNormalizedArielDataset", 
                "p_StandardNormal", 
                "m_plain")
            )[:args.top_k]
        )
    
    best_runs.append(
        get_best_runs(
            os.path.join(
                args.runs_dir, 
                "dt_RealNormalizedArielDataset", 
                "p_StandardNormal", 
                "m_autoencoder")
            )[:args.top_k]
        )

    best_runs.append(
        get_best_runs(
            os.path.join(
                args.runs_dir, 
                "dt_RealStandardizedArielDataset", 
                "p_StandardNormal", 
                "m_plain")
            )[:args.top_k]
        )
    
    best_runs.append(
        get_best_runs(
            os.path.join(
                args.runs_dir, 
                "dt_RealStandardizedArielDataset", 
                "p_StandardNormal", 
                "m_autoencoder")
            )[:args.top_k]
        )

    best_runs.append(
        get_best_runs(
            os.path.join(
                args.runs_dir, 
                "dt_NormalizedArielDataset", 
                "p_StandardNormal", 
                "m_plain")
            )[:args.top_k]
        )
    
    best_runs.append(
        get_best_runs(
            os.path.join(
                args.runs_dir, 
                "dt_NormalizedArielDataset", 
                "p_StandardNormal", 
                "m_autoencoder")
            )[:args.top_k]
        )
    
    best_runs.append(
        get_best_runs(
            os.path.join(
                args.runs_dir,
                "dt_NoisyNormalizedArielDataset",
                "p_StandardNormal",
                "m_plain")
            )[:args.top_k]
        )
    
    best_runs.append(
        get_best_runs(
            os.path.join(
                args.runs_dir,
                "dt_NoisyNormalizedArielDataset",
                "p_StandardNormal",
                "m_autoencoder")
            )[:args.top_k]
        )
    
    best_runs.append(
        get_best_runs(
            os.path.join(
                args.runs_dir,
                "dt_NoisyStandardizedArielDataset",
                "p_StandardNormal",
                "m_plain")
            )[:args.top_k]
        )
    
    best_runs.append(
        get_best_runs(
            os.path.join(
                args.runs_dir,
                "dt_NoisyStandardizedArielDataset",
                "p_StandardNormal",
                "m_autoencoder")
            )[:args.top_k]
        )
    
    best_runs.append(
        get_best_runs(
            os.path.join(
                args.runs_dir,
                "dt_RealNoisyNormalizedArielDataset",
                "p_StandardNormal",
                "m_plain")
            )[:args.top_k]
        )


    best_runs.append(
        get_best_runs(
            os.path.join(
                args.runs_dir,
                "dt_RealNoisyNormalizedArielDataset",
                "p_StandardNormal",
                "m_autoencoder")
            )[:args.top_k]
        )
    
    best_runs.append(
        get_best_runs(
            os.path.join(
                args.runs_dir,
                "dt_RealNoisyStandardizedArielDataset",
                "p_StandardNormal",
                "m_plain")
            )[:args.top_k]
        )
    
    best_runs.append(
        get_best_runs(
            os.path.join(
                args.runs_dir,
                "dt_RealNoisyStandardizedArielDataset",
                "p_StandardNormal",
                "m_autoencoder")
            )[:args.top_k]
        )
    
    
    output_fname = f"best_runs_top{args.top_k}.csv"
    print(f"Saving best runs to {output_fname}")
    datasets = np.array([dt for br in best_runs for dt, _, _, _ in br]).reshape(1, -1)
    runs = np.array([run for br in best_runs for _, run, _, _ in br]).reshape(1, -1)
    epochs = np.array([epoch for br in best_runs for _, _, epoch, _ in br]).reshape(1, -1)
    losses = np.array([loss for br in best_runs for _, _, _, loss in br]).reshape(1, -1)

    np_df = np.concatenate([datasets, runs, epochs, losses], axis=0).T

    df = pd.DataFrame(np_df, columns=["Dataset", "Run", "Epoch", "Loss"])
    df.to_csv(os.path.join(args.runs_dir, output_fname), index=False)
    show_best_runs(best_runs, args.top_k)
    