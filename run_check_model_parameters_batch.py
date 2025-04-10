import os
from check_model_parameters import check_model_parameters
import numpy as np
import argparse
import yaml
from tqdm import tqdm
from pprint import pprint

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Check model's parameters")
    ap.add_argument("--runs_dir", type=str, required=True, help="Path to the runs directory")
    ap.add_argument("--filters", type=str, nargs='+', default=[], help="Filters to apply to the runs directory")
    args = ap.parse_args()
    
    
    # get all settings files in the runs directory
    settings_files = []
    for root, dirs, files in os.walk(args.runs_dir):
        for file in files:
            if file.endswith(".yaml"):
                path = os.path.join(root, file)
                # apply filters
                if args.filters:
                    flag = True
                    for filter in args.filters:
                        if filter not in path:
                            flag = False
                            break
                    if flag:
                        settings_files.append(path)
                else:
                    # no filters, add all settings files
                    settings_files.append(path)
    
    # check model parameters for each settings file
    total_num_model_params = []
    for settings_file in tqdm(settings_files, desc="Checking model parameters"):
        # load settings file
        with open(settings_file, "r") as f:
            settings = yaml.safe_load(f)
            
        settings['training']['device'] = 'cpu'
        
        # check model parameters
        num_params = check_model_parameters(settings)
        total_num_model_params.append(sum([v for k, v in num_params.items()]))
        
        
    print("Minimum number of model parameters: ", min(total_num_model_params), " => ", settings_files[np.argmin(total_num_model_params)])
    print("Maximum number of model parameters: ", max(total_num_model_params), " => ", settings_files[np.argmax(total_num_model_params)])
    print("Average number of model parameters: ", sum(total_num_model_params)/len(total_num_model_params))
    print("Standard deviation of number of model parameters: ", np.std(total_num_model_params))
    print("Number of runs: ", len(total_num_model_params))
        
        