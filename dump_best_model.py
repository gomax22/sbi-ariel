
import argparse
import os
from pathlib import Path
import torch
import yaml
from pprint import pprint
from tqdm import tqdm

basedir = Path(__file__).resolve().parent


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train a model")
    ap.add_argument("--runs_dir", type=str, required=True, help="Path to the runs directory")
    args = ap.parse_args()

    # scan recursively the runs directory
    not_found = 0
    total = sum([1 for r, d, folder in os.walk(args.runs_dir) for f in folder if f == "settings.yaml"])
    with tqdm(total=total, desc="Writing metadatas...") as pbar:
        for root, dirs, files in os.walk(args.runs_dir):
            for file in files:
                if file != "settings.yaml":
                    continue
                
                # load settings
                with open(os.path.join(root, file), "r") as f:
                    settings = yaml.safe_load(f)
                
                if "metadata" in settings and "best_model_epoch" in settings["metadata"]:
                    pbar.update(1)
                    continue

                # load the best model
                try:
                    best_model = torch.load(os.path.join(root, "best_model.pt"), map_location=settings["training"].get("device", "cpu"), weights_only=True)
                except FileNotFoundError:
                    print(f"Warning: best model not found in {root}")
                    not_found += 1
                    pbar.update(1)
                    continue

                # recover settings
                real = "real_" if "real" in settings["dataset"]["type"].lower() else ""
                context = "full_" if "full" in settings["dataset"]["type"].lower() else ""
                prior = "standardized_" if settings["model"]["prior"]["type"] == "StandardNormal" else "normalized_"
                theta_emb = "embedding_theta_" if "theta_embedding_kwargs" in settings["model"]["posterior_kwargs"] else ""
                
                settings_fname = f"{real}{context}{prior}{theta_emb}settings.yaml"

                # find absolute path to the settings file from basedir directory
                settings_fpath = os.path.abspath(os.path.join(basedir, "settings", settings_fname))

                # check existence of the settings file
                if not os.path.exists(settings_fpath):
                    raise FileNotFoundError(f"Settings file {settings_fpath} not found")
                
                # update settings with metadata
                settings["metadata"] = {
                    "settings_file": settings_fpath,
                    "file": os.path.abspath(__file__), # tell us if the run was completed or not
                    "best_model_epoch": best_model["epoch"],
                }

                # dump settings after training
                with open(os.path.join(root, file), "w") as f:
                    yaml.dump(settings, f)

                pbar.update(1)
    print(f"Warning: {not_found} best models not found")

