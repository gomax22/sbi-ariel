import argparse
import os
import yaml

import yaml
from dingo.core.posterior_models.build_model import ( # type: ignore
    build_model_from_kwargs,
    autocomplete_model_kwargs,
)

def check_model_parameters(
    settings: dict,
):
    """
    Check the model's parameters.
    
    Args:
        settings (dict): Settings for the model.
        input_dim (int): Input dimension.
        context_dim (int): Context dimension.
        
    Returns:
        None
    """
    autocomplete_model_kwargs(
        settings["model"],
        input_dim=settings["task"]["dim_theta"],  # input = theta dimension
        context_dim=settings["task"]["dim_x"],  # context dim = observation dimension
    )

    
    model = build_model_from_kwargs(
            settings={"train_settings": settings},
            device=settings["training"].get("device", "cpu"),
        )
    
    
    # get number of parameters of the model 
    num_params = {
        "continuous_flow": sum([p.numel() for p in model.network.continuous_flow.parameters()]),
        "theta_embedding_net": sum([p.numel() for p in model.network.theta_embedding_net.parameters()]),
        "context_embedding_net": sum([p.numel() for p in model.network.context_embedding_net.parameters()])
    }
    return num_params


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description="Check model's parameters")
    ap.add_argument("--run_dir", type=str, required=True, help="Path to the settings file")
    args = ap.parse_args()
    
    # load settings
    with open(os.path.join(args.run_dir, "settings.yaml"), "r") as f:
        settings = yaml.safe_load(f)
    
    num_params = check_model_parameters(settings)
    
    for k, v in num_params.items():
        print(f"{k}: {v:,} (size: {v * 4 / 1024 / 1024:.2f} MB)")
    