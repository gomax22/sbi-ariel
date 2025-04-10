import numpy as np
import torch
import os
import pandas as pd
import h5py
from pathlib import Path
from argparse import ArgumentParser
import random

from utils import to_observed_matrix, standardize, normalize, augment_data_with_noise

pd.options.mode.chained_assignment = None
RJUP = 69911000
MJUP = 1.898e27
RSOL = 696340000

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
random.seed(SEED)


def make_dataset(training_path: str, training_gt_path: str, test_path: str | None, output_path: str):
    
    # Load the training data
    spectral_training_data = h5py.File(os.path.join(training_path,'SpectralData.hdf5'),"r")
    aux_training_data = pd.read_csv(os.path.join(training_path,'AuxillaryTable.csv'))
    soft_label_data = pd.read_csv(os.path.join(training_gt_path, 'FM_Parameter_Table.csv'))
    infile = h5py.File(os.path.join(training_gt_path, 'Tracedata.hdf5'), "r")
    
    # returns N x 52 x 4 matrix, where N is the number of samples, 52 is the number of channels, and 4 are the observational data
    #  0: wavelength grid, 1: observed spectrum, 2: noise, 3: wavelength width
    # We can safely discard wlgrid (wavelength grid) and wlwidth (width of wavelength) since they are unchanged in the dataset
    spec_matrix = to_observed_matrix(spectral_training_data, aux_training_data)
    spectra, noises = spec_matrix[:, :, 1], spec_matrix[:, :, 2]
    print(f"Spectra shape: {spectra.shape}\nNoises shape: {noises.shape}")

    aux_training_data['star_radius_m'] = aux_training_data['star_radius_m'] / RSOL
    aux_training_data['planet_mass_kg'] = aux_training_data['planet_mass_kg'] / MJUP
    
    # Filter out the planets that do not have tracedata
    # Planetlist from tracedata ==> 'Planet_train1', 'Planet_train10', 'Planet_train100', 'Planet_train1000', etc.
    # Planetlist from aux, fm ==> 'train1', 'train10', 'train100', 'train1000', etc.
    planetlist = sorted([p for p in infile.keys()])
    train_planet_ids = sorted([p[7:] for p in planetlist if not infile[p]['weights'].shape])  #Planet_train -> train
    train_indices = [int(p[12:]) - 1  for p in planetlist if not infile[p]['weights'].shape]
    
    planet_ids = [p[7:] for p in planetlist if infile[p]['weights'].shape]

    # 0.1 * 0.24 = 0.024 of the data is used for validation (test)
    test_planet_ids = sorted(random.sample(planet_ids, k=int(len(planet_ids) * 0.1)))
    test_indices = [int(p[12:]) - 1  for p in planetlist if p[7:] in test_planet_ids]
    
    valid_planet_ids = sorted([p for p in planet_ids if p not in test_planet_ids])
    valid_indices = [int(p[12:]) - 1  for p in planetlist if p[7:] in valid_planet_ids]

    if not Path(output_path).exists():
        Path(output_path).mkdir(parents=True, exist_ok=True)
    np.save(os.path.join(output_path, "valid_mapping.npy"), valid_indices) 
    np.save(os.path.join(output_path, "test_mapping.npy"), test_indices)

    
    aux_training_data_array = np.array(aux_training_data.iloc[:, 1:], dtype=np.float32)
    soft_label_data_array = np.array(soft_label_data.iloc[:, 2:], dtype=np.float32)


    train_aux_training_data = aux_training_data_array[train_indices]
    train_targets = soft_label_data_array[train_indices]
    train_spectra = spectra[train_indices]
    train_noises = noises[train_indices]

    valid_aux_training_data = aux_training_data_array[valid_indices]
    valid_targets = soft_label_data_array[valid_indices]
    valid_spectra = spectra[valid_indices]
    valid_noises = noises[valid_indices]

    test_aux_training_data = aux_training_data_array[test_indices]
    test_targets = soft_label_data_array[test_indices]
    test_spectra = spectra[test_indices]
    test_noises = noises[test_indices]
    
    # Sample the noise and add it (maybe we need to augment the data)
    train_real_spectra = np.random.normal(loc=train_spectra, scale=train_noises, size=train_spectra.shape)
    valid_real_spectra = np.random.normal(loc=valid_spectra, scale=valid_noises, size=valid_spectra.shape)
    test_real_spectra = np.random.normal(loc=test_spectra, scale=test_noises, size=test_spectra.shape)
    
    np.save(os.path.join(output_path, 'train_ideal_spectra.npy'), train_spectra)
    np.save(os.path.join(output_path, 'train_noises.npy'), train_noises)
    np.save(os.path.join(output_path, 'train_aux_data.npy'), train_aux_training_data)
    np.save(os.path.join(output_path, 'train_targets.npy'), train_targets)
    np.save(os.path.join(output_path, 'train_real_spectra.npy'), train_real_spectra)

    np.save(os.path.join(output_path, 'valid_ideal_spectra.npy'), valid_spectra)
    np.save(os.path.join(output_path, 'valid_noises.npy'), valid_noises)
    np.save(os.path.join(output_path, 'valid_aux_data.npy'), valid_aux_training_data)
    np.save(os.path.join(output_path, 'valid_targets.npy'), valid_targets)
    np.save(os.path.join(output_path, 'valid_real_spectra.npy'), valid_real_spectra)
    np.save(os.path.join(output_path, 'valid_mapping.npy'), valid_indices)

    np.save(os.path.join(output_path, 'test_mapping.npy'), test_indices)
    np.save(os.path.join(output_path, 'test_ideal_spectra.npy'), test_spectra)
    np.save(os.path.join(output_path, 'test_noises.npy'), test_noises)
    np.save(os.path.join(output_path, 'test_aux_data.npy'), test_aux_training_data)
    np.save(os.path.join(output_path, 'test_targets.npy'), test_targets)
    np.save(os.path.join(output_path, 'test_real_spectra.npy'), test_real_spectra)

    print(f"Training ideal data shape: {train_spectra.shape}")
    print(f"Training noise data shape: {train_noises.shape}")
    print(f"Training aux data shape: {train_aux_training_data.shape}")
    print(f"Training target data shape: {train_targets.shape}")
    print(f"Training real data shape: {train_real_spectra.shape}")
    
    print(f"Validation ideal data shape: {valid_spectra.shape}")
    print(f"Validation noise data shape: {valid_noises.shape}")
    print(f"Validation aux data shape: {valid_aux_training_data.shape}")
    print(f"Validation target data shape: {valid_targets.shape}")
    print(f"Validation real data shape: {valid_real_spectra.shape}")
    print(f"Validation mapping shape: {len(valid_indices)}")

    print(f"Test ideal data shape: {test_spectra.shape}")
    print(f"Test noise data shape: {test_noises.shape}")
    print(f"Test aux data shape: {test_aux_training_data.shape}")
    print(f"Test target data shape: {test_targets.shape}")
    print(f"Test real data shape: {test_real_spectra.shape}")
    print(f"Test mapping shape: {len(test_indices)}")
        
    # test dataset
    # Load the training data
    if test_path is None:
        return
    spectral_test_data = h5py.File(os.path.join(test_path,'SpectralData.hdf5'),"r")
    aux_test_data = pd.read_csv(os.path.join(test_path,'AuxillaryTable.csv'))

    test_spec_matrix = to_observed_matrix(spectral_test_data, aux_test_data)
    test_spectra, test_noises = test_spec_matrix[:, :, 1], test_spec_matrix[:, :, 2]
    print(f"Test spectra shape: {test_spectra.shape}\nTest Noises shape: {test_noises.shape}")

    aux_test_data['star_radius_m'] = aux_test_data['star_radius_m'] / RSOL
    aux_test_data['planet_mass_kg'] = aux_test_data['planet_mass_kg'] / MJUP

    aux_test_data = np.array(aux_test_data.iloc[:, 1:], dtype=np.float32)

    np.save(os.path.join(output_path, 'effective_test_aux_data.npy'), aux_test_data)
    np.save(os.path.join(output_path, 'effective_test_noises.npy'), test_noises)
    np.save(os.path.join(output_path, 'effective_test_ideal_spectra.npy'), test_spectra)

    # augment test spectra too
    # they should be aligned. If not, we have a problem
    test_noise_prof = np.random.normal(loc=0, scale=test_noises, size=test_noises.shape)
    test_real_spectra = test_spectra + test_noise_prof

    np.save(os.path.join(output_path, 'effective_test_real_spectra.npy'), test_real_spectra)
    np.save(os.path.join(output_path, 'effective_test_noise_profiles.npy'), test_noise_prof)

    print(f"Test ideal data shape: {test_spectra.shape}")
    print(f"Test noise data shape: {test_noises.shape}")
    print(f"Test aux data shape: {aux_test_data.shape}")
    print(f"Test real data shape: {test_real_spectra.shape}")
    print(f"Test noise profile data shape: {test_noise_prof.shape}")



if __name__ == '__main__':
    ap = ArgumentParser(description='Make the ADC dataset suitable for training')
    ap.add_argument('--training_path', type=str, required=True, help='Path to the training data')
    ap.add_argument('--training_gt_path', type=str, required=True, help='Path to the training ground truth data')
    ap.add_argument('--test_path', type=str, default=None, help='Path to the test data')
    ap.add_argument('--output_path', type=str, required=True, help='Path to save the dataset')
    args = ap.parse_args()
    args = vars(args)
    make_dataset(**args)

