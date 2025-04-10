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

                best_runs.append((dataset, root, best_model_epoch, history[best_model_epoch-1,2]))
                pbar.update(1)
    
    best_runs = sorted(best_runs, key=lambda x: x[3])
    return best_runs



def show_best_runs(best_runs, top_k):
    print("Best runs:")
    for br in best_runs:
        for dt, run, epoch, loss in br[:top_k]:
            print(f"Dataset: {dt}, Run: {f'{os.sep}'.join(run.split(os.sep)[5:])}, Epoch: {epoch}, Loss: {loss}")


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
    



    
"""
last runs:
Run: last_runs/p_StandardNormal/ctxt_full/m_autoencoder/theta_emb_0/layers_14/dropout_0.1/a_-2.0/lr_0.0005/20241207_161243, Epoch: 433, Loss: 0.2383539389464291
Run: last_runs/p_StandardNormal/ctxt_full/m_plain/theta_emb_0/layers_7/dropout_0.1/a_-2.0/lr_0.0005/20241208_014354, Epoch: 139, Loss: 0.2329077216137613
Run: last_runs/p_StandardNormal/ctxt_aux_data/m_autoencoder/theta_emb_0/layers_7/dropout_0.0/a_-2.0/lr_0.0005/20241207_160701, Epoch: 444, Loss: 0.2474568630545834
Run: last_runs/p_StandardNormal/ctxt_aux_data/m_plain/theta_emb_0/layers_7/dropout_0.0/a_-2.0/lr_0.0005/20241208_015818, Epoch: 169, Loss: 0.2392382429029803
Run: last_runs/p_Uniform/ctxt_full/m_autoencoder/theta_emb_0/layers_7/dropout_0.0/a_-2.0/lr_0.0005/20241207_160645, Epoch: 402, Loss: 0.0190985113587065
Run: last_runs/p_Uniform/ctxt_full/m_plain/theta_emb_0/layers_7/dropout_0.0/a_-2.0/lr_0.0005/20241208_015712, Epoch: 129, Loss: 0.0184450719936936
Run: last_runs/p_Uniform/ctxt_aux_data/m_autoencoder/theta_emb_0/layers_7/dropout_0.1/a_-2.0/lr_0.0005/20241207_140719, Epoch: 416, Loss: 0.0184864342391812
Run: last_runs/p_Uniform/ctxt_aux_data/m_plain/theta_emb_0/layers_14/dropout_0.0/a_-2.0/lr_0.0005/20241208_131225, Epoch: 144, Loss: 0.0185269182392554
"""

"""
Last runs (a > -0.5):
Run: last_runs/p_StandardNormal/ctxt_full/m_autoencoder/theta_emb_0/layers_14/dropout_0.1/a_-0.5/lr_0.0005/20241207_201549, Epoch: 163, Loss: 1.0352036772813134
Run: last_runs/p_StandardNormal/ctxt_full/m_plain/theta_emb_0/layers_7/dropout_0.1/a_-0.5/lr_0.0005/20241208_014544, Epoch: 163, Loss: 1.0344161731104151
Run: last_runs/p_StandardNormal/ctxt_aux_data/m_autoencoder/theta_emb_0/layers_7/dropout_0.0/a_-0.5/lr_0.0005/20241208_225933, Epoch: 167, Loss: 1.0349200223449169
Run: last_runs/p_StandardNormal/ctxt_aux_data/m_plain/theta_emb_0/layers_14/dropout_0.1/a_-0.5/lr_0.0005/20241208_094549, Epoch: 160, Loss: 1.0335544653097946
Run: last_runs/p_Uniform/ctxt_full/m_autoencoder/theta_emb_0/layers_14/dropout_0.0/a_-0.5/lr_0.0005/20241207_183635, Epoch: 418, Loss: 0.0791770700690673
Run: last_runs/p_Uniform/ctxt_full/m_plain/theta_emb_0/layers_7/dropout_0.0/a_-0.5/lr_0.0005/20241208_093557, Epoch: 162, Loss: 0.0795793575221001
Run: last_runs/p_Uniform/ctxt_aux_data/m_autoencoder/theta_emb_0/layers_7/dropout_0.1/a_-0.5/lr_0.0005/20241208_225837, Epoch: 150, Loss: 0.0796232874613982
Run: last_runs/p_Uniform/ctxt_aux_data/m_plain/theta_emb_0/layers_14/dropout_0.0/a_-0.5/lr_0.0005/20241208_131426, Epoch: 156, Loss: 0.080314422810452
"""

"""
Best runs:
Dataset: dt_FullNormalizedArielDataset, Run: layers_7/mfl_1/dropout_0.1/a_-0.75/lr_0.0001/20250220_111529, Epoch: 117, Loss: 0.0376813038964266
Dataset: dt_FullNormalizedArielDataset, Run: layers_14/mfl_2/dropout_0.0/a_-0.75/lr_0.0001/20250220_003610, Epoch: 205, Loss: 0.0398934565235927
Dataset: dt_FullStandardizedArielDataset, Run: layers_14/mfl_2/dropout_0.1/a_-0.75/lr_0.0001/20250220_114102, Epoch: 449, Loss: 0.1667498033841061
Dataset: dt_FullStandardizedArielDataset, Run: layers_7/mfl_1/dropout_0.1/a_-0.75/lr_0.0001/20250220_113323, Epoch: 166, Loss: 0.1587740229378738
Dataset: dt_RealFullNormalizedArielDataset, Run: layers_7/mfl_1/dropout_0.1/a_-0.75/lr_0.0001/20250221_083207, Epoch: 147, Loss: 0.0475820898059188
Dataset: dt_RealFullNormalizedArielDataset, Run: layers_7/mfl_1/dropout_0.1/a_-0.75/lr_0.0001/20250221_081629, Epoch: 149, Loss: 0.0479838618838664
Dataset: dt_RealNormalizedArielDataset, Run: layers_7/mfl_1/dropout_0.1/a_-0.75/lr_0.0001/20250222_004230, Epoch: 147, Loss: 0.0482105681923116
Dataset: dt_RealNormalizedArielDataset, Run: layers_7/mfl_1/dropout_0.0/a_-0.75/lr_0.0001/20250221_181340, Epoch: 144, Loss: 0.0468892504016012
"""