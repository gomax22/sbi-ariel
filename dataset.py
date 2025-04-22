from bdb import effective
import numpy as np
import torch
from os.path import join
from torch.utils.data import Dataset
from pyro import distributions as pdist
import pyro
from typing import Callable


class ArielDataset(Dataset):
    def __init__(self):
        super(ArielDataset, self).__init__()

        self.prior_params = {
            "low": torch.tensor([0.1, 0.0, -12.0, -12.0, -12.0, -12.0, -12.0]),
            "high": torch.tensor([3.0, 7000.0, -1.0, -1.0, -1.0, -1.0, -1.0]),
        }

        self.prior_dist = pdist.Uniform(**self.prior_params).to_event(1)
        self.prior_dist.set_default_validate_args(False)

    def get_prior(self) -> Callable:
        def prior(num_samples=1):
            return pyro.sample("parameters", self.prior_dist.expand_by([num_samples]))

        return prior
    
    # self.preprocessing not instantiated here, but in the child classes
    def standardize(self, sample, label, inverse=False):
        mean = self.preprocessing[label]["mean"]
        std = self.preprocessing[label]["std"]
        if not inverse:
            return (sample - mean) / std
        else:
            return (sample * std) + mean    
        
    # self.preprocessing not instantiated here, but in the child classes
    def normalize(self, sample, label, inverse=False):
        min_ = self.preprocessing[label]["min"]
        max_ = self.preprocessing[label]["max"]
        if not inverse:
            return (sample - min_) / (max_ - min_)
        else:
            return sample * (max_ - min_) + min_
    
class StandardizedArielDataset(ArielDataset):
    def __init__(self, theta, spectra, preprocessing=None):
        super(StandardizedArielDataset, self).__init__()

        self.preprocessing = {
            "spectra": {"min": torch.min(spectra, dim=0)[0], "max": torch.max(spectra, dim=0)[0]},
            "theta": {"mean": torch.mean(theta, dim=0), "std": torch.std(theta, dim=0)},
        } if preprocessing is None else preprocessing

        self.theta = self.standardize(theta, "theta")
        self.x = self.normalize(spectra, "spectra")

    def __len__(self):
        return len(self.theta)
    
    def __getitem__(self, idx):
        return self.theta[idx], self.x[idx]
    
class FullStandardizedArielDataset(ArielDataset):
    def __init__(self, theta, spectra, noises, aux_data, preprocessing=None):
        super(FullStandardizedArielDataset, self).__init__()

        self.preprocessing = {
            "spectra": {"min": torch.min(spectra, dim=0)[0], "max": torch.std(spectra, dim=0)},
            "noises": {"min": torch.min(noises, dim=0)[0], "max": torch.max(noises, dim=0)[0]},
            "aux_data": {"min": torch.min(aux_data, dim=0)[0], "max": torch.max(aux_data, dim=0)[0]},
            "theta": {"mean": torch.mean(theta, dim=0), "std": torch.std(theta, dim=0)},
        } if preprocessing is None else preprocessing

        self.theta = self.standardize(theta, "theta")
        self.spectra = self.normalize(spectra, "spectra")
        self.noises = self.normalize(noises, "noises")
        self.aux_data = self.normalize(aux_data, "aux_data")

        # not the best way of conditioning so far...
        self.x = torch.cat([self.spectra, self.noises, self.aux_data], dim=1)

    def __len__(self):
        return len(self.theta)

    def __getitem__(self, idx):
        return self.theta[idx], self.x[idx]

class NormalizedArielDataset(ArielDataset):
    def __init__(self, theta, spectra, preprocessing=None):
        super(NormalizedArielDataset, self).__init__()

        self.preprocessing = {
            "spectra": {"min": torch.min(spectra, dim=0)[0], "max": torch.max(spectra, dim=0)[0]},
            "theta": {"min": torch.min(theta, dim=0)[0], "max": torch.max(theta, dim=0)[0]},
        } if preprocessing is None else preprocessing

        self.theta = self.normalize(theta, "theta")
        self.x = self.normalize(spectra, "spectra")

        
    def __len__(self):
        return len(self.theta)
    
    def __getitem__(self, idx):
        return self.theta[idx], self.x[idx]


class FullNormalizedArielDataset(ArielDataset):
    def __init__(self, theta, spectra, noises, aux_data, preprocessing=None):
        super(FullNormalizedArielDataset, self).__init__()

        self.preprocessing = {
            "spectra": {"min": torch.min(spectra, dim=0)[0], "max": torch.std(spectra, dim=0)},
            "noises": {"min": torch.min(noises, dim=0)[0], "max": torch.max(noises, dim=0)[0]},
            "aux_data": {"min": torch.min(aux_data, dim=0)[0], "max": torch.max(aux_data, dim=0)[0]},
            "theta": {"min": torch.min(theta, dim=0)[0], "max": torch.max(theta, dim=0)[0]},
        } if preprocessing is None else preprocessing

        self.theta = self.normalize(theta, "theta")
        self.spectra = self.normalize(spectra, "spectra") # ideal
        self.noises = self.normalize(noises, "noises") 
        self.aux_data = self.normalize(aux_data, "aux_data")

        # not the best way of conditioning so far
        self.x = torch.cat([self.spectra, self.noises, self.aux_data], dim=1)

    def __len__(self):
        return len(self.theta)

    def __getitem__(self, idx):
        return self.theta[idx], self.x[idx]


class NoisyNormalizedArielDataset(ArielDataset):
    def __init__(self, theta, spectra, noises, preprocessing=None):
        super(NoisyNormalizedArielDataset, self).__init__()

        self.preprocessing = {
            "spectra": {"min": torch.min(spectra, dim=0)[0], "max": torch.std(spectra, dim=0)},
            "noises": {"min": torch.min(noises, dim=0)[0], "max": torch.max(noises, dim=0)[0]},
            "theta": {"min": torch.min(theta, dim=0)[0], "max": torch.max(theta, dim=0)[0]},
        } if preprocessing is None else preprocessing

        self.theta = self.normalize(theta, "theta")
        self.spectra = self.normalize(spectra, "spectra") # real
        self.noises = self.normalize(noises, "noises")

        # not the best way of conditioning so far
        self.x = torch.cat([self.spectra, self.noises], dim=1)

    def __len__(self):
        return len(self.theta)

    def __getitem__(self, idx):
        return self.theta[idx], self.x[idx]


class NoisyStandardizedArielDataset(ArielDataset):
    def __init__(self, theta, spectra, noises, preprocessing=None):
        super(NoisyStandardizedArielDataset, self).__init__()

        self.preprocessing = {
            "spectra": {"min": torch.min(spectra, dim=0)[0], "max": torch.std(spectra, dim=0)},
            "noises": {"min": torch.min(noises, dim=0)[0], "max": torch.max(noises, dim=0)[0]},
            "theta": {"mean": torch.mean(theta, dim=0), "std": torch.std(theta, dim=0)},
        } if preprocessing is None else preprocessing

        self.theta = self.standardize(theta, "theta")
        self.spectra = self.normalize(spectra, "spectra") # real
        self.noises = self.normalize(noises, "noises")

        # not the best way of conditioning so far
        self.x = torch.cat([self.spectra, self.noises], dim=1)

    def __len__(self):
        return len(self.theta)

    def __getitem__(self, idx):
        return self.theta[idx], self.x[idx]



class RealNormalizedArielDataset(ArielDataset):
    def __init__(self, theta, spectra, aux_data, preprocessing=None):
        super(RealNormalizedArielDataset, self).__init__()

        self.preprocessing = {
            "spectra": {"min": torch.min(spectra, dim=0)[0], "max": torch.std(spectra, dim=0)},
            "aux_data": {"min": torch.min(aux_data, dim=0)[0], "max": torch.max(aux_data, dim=0)[0]},
            "theta": {"min": torch.min(theta, dim=0)[0], "max": torch.max(theta, dim=0)[0]},
        } if preprocessing is None else preprocessing

        self.theta = self.normalize(theta, "theta")
        self.spectra = self.normalize(spectra, "spectra") # real
        self.aux_data = self.normalize(aux_data, "aux_data")

        # not the best way of conditioning so far
        self.x = torch.cat([self.spectra, self.aux_data], dim=1)

    def __len__(self):
        return len(self.theta)

    def __getitem__(self, idx):
        return self.theta[idx], self.x[idx]

    
class RealStandardizedArielDataset(ArielDataset):
    def __init__(self, theta, spectra, aux_data, preprocessing=None):
        super(RealStandardizedArielDataset, self).__init__()

        self.preprocessing = {
            "spectra": {"min": torch.min(spectra, dim=0)[0], "max": torch.std(spectra, dim=0)},
            "aux_data": {"min": torch.min(aux_data, dim=0)[0], "max": torch.max(aux_data, dim=0)[0]},
            "theta": {"mean": torch.mean(theta, dim=0), "std": torch.std(theta, dim=0)},
        } if preprocessing is None else preprocessing

        self.theta = self.standardize(theta, "theta")
        self.spectra = self.normalize(spectra, "spectra")
        self.aux_data = self.normalize(aux_data, "aux_data")

        # not the best way of conditioning so far...
        self.x = torch.cat([self.spectra, self.aux_data], dim=1)

    def __len__(self):
        return len(self.theta)

    def __getitem__(self, idx):
        return self.theta[idx], self.x[idx]


def load_normalized_ariel_dataset(settings):
    directory_save = settings["dataset"]["path"]
    train_targets = torch.tensor(np.load(join(directory_save, 'train_targets.npy')), dtype=torch.float)
    train_spectra =  torch.tensor(np.load(join(directory_save, 'train_real_spectra.npy')), dtype=torch.float)

    valid_spectra = torch.tensor(np.load(join(directory_save, 'valid_real_spectra.npy')), dtype=torch.float)
    valid_targets = torch.tensor(np.load(join(directory_save, 'valid_targets.npy')), dtype=torch.float) 
    
    test_spectra = torch.tensor(np.load(join(directory_save, 'test_real_spectra.npy')), dtype=torch.float)
    test_targets = torch.tensor(np.load(join(directory_save, 'test_targets.npy')), dtype=torch.float)
    

    train_dataset = NormalizedArielDataset(
        torch.cat([train_targets, valid_targets], dim=0),
        torch.cat([train_spectra, valid_spectra], dim=0)
    )

    settings["task"]["dim_theta"] = train_targets.shape[1]
    settings["task"]["dim_x"] = train_spectra.shape[1]    
    
    valid_dataset = NormalizedArielDataset(test_targets, test_spectra, preprocessing=train_dataset.preprocessing)
    
    effective_test_spectra = torch.tensor(np.load(join(directory_save, 'effective_test_real_spectra.npy')), dtype=torch.float)
    fake_targets = torch.ones((len(effective_test_spectra), train_targets.shape[1]), dtype=torch.float)
    test_dataset = NormalizedArielDataset(fake_targets, effective_test_spectra, preprocessing=train_dataset.preprocessing)
    
    return train_dataset, valid_dataset, test_dataset

def load_full_normalized_ariel_dataset(settings):
    directory_save = settings["dataset"]["path"]
    train_targets = torch.tensor(np.load(join(directory_save, 'train_targets.npy')), dtype=torch.float)
    train_spectra =  torch.tensor(np.load(join(directory_save, 'train_ideal_spectra.npy')), dtype=torch.float)
    train_noises = torch.tensor(np.load(join(directory_save, 'train_noises.npy')), dtype=torch.float)
    train_aux_data = torch.tensor(np.load(join(directory_save, 'train_aux_data.npy')), dtype=torch.float)
    
    valid_spectra = torch.tensor(np.load(join(directory_save, 'valid_ideal_spectra.npy')), dtype=torch.float)
    valid_targets = torch.tensor(np.load(join(directory_save, 'valid_targets.npy')), dtype=torch.float)
    valid_noises = torch.tensor(np.load(join(directory_save, 'valid_noises.npy')), dtype=torch.float)
    valid_aux_data = torch.tensor(np.load(join(directory_save, 'valid_aux_data.npy')), dtype=torch.float)
    
    test_spectra = torch.tensor(np.load(join(directory_save, 'test_ideal_spectra.npy')), dtype=torch.float)
    test_noises = torch.tensor(np.load(join(directory_save, 'test_noises.npy')), dtype=torch.float)
    test_aux_data = torch.tensor(np.load(join(directory_save, 'test_aux_data.npy')), dtype=torch.float)
    test_targets = torch.tensor(np.load(join(directory_save, 'test_targets.npy')), dtype=torch.float)

    train_dataset = FullNormalizedArielDataset(
        torch.cat([train_targets, valid_targets], dim=0),
        torch.cat([train_spectra, valid_spectra], dim=0),
        torch.cat([train_noises, valid_noises], dim=0),
        torch.cat([train_aux_data, valid_aux_data], dim=0)
    )

    settings["task"]["dim_theta"] = train_targets.shape[1]
    settings["task"]["dim_x"] = train_spectra.shape[1] + train_noises.shape[1] + train_aux_data.shape[1]
    
    valid_dataset = FullNormalizedArielDataset(test_targets, test_spectra, test_noises, test_aux_data, preprocessing=train_dataset.preprocessing)
    
    effective_test_spectra = torch.tensor(np.load(join(directory_save, 'effective_test_ideal_spectra.npy')), dtype=torch.float)
    effective_test_noises = torch.tensor(np.load(join(directory_save, 'effective_test_noises.npy')), dtype=torch.float)
    effective_test_aux_data = torch.tensor(np.load(join(directory_save, 'effective_test_aux_data.npy')), dtype=torch.float)
    fake_targets = torch.ones((len(effective_test_spectra), train_targets.shape[1]), dtype=torch.float)
    test_dataset = FullNormalizedArielDataset(fake_targets, effective_test_spectra, effective_test_noises, effective_test_aux_data, preprocessing=train_dataset.preprocessing)
    
    return train_dataset, valid_dataset, test_dataset

def load_standardized_ariel_dataset(settings):
    directory_save = settings["dataset"]["path"]
    train_targets = torch.tensor(np.load(join(directory_save, 'train_targets.npy')), dtype=torch.float)
    train_spectra =  torch.tensor(np.load(join(directory_save, 'train_real_spectra.npy')), dtype=torch.float)

    valid_spectra = torch.tensor(np.load(join(directory_save, 'valid_real_spectra.npy')), dtype=torch.float)
    valid_targets = torch.tensor(np.load(join(directory_save, 'valid_targets.npy')), dtype=torch.float)  
    
    test_spectra = torch.tensor(np.load(join(directory_save, 'test_real_spectra.npy')), dtype=torch.float)
    test_targets = torch.tensor(np.load(join(directory_save, 'test_targets.npy')), dtype=torch.float)
    
    
    train_dataset = StandardizedArielDataset(
        torch.cat([train_targets, valid_targets], dim=0),
        torch.cat([train_spectra, valid_spectra], dim=0)
    )

    settings["task"]["dim_theta"] = train_targets.shape[1]
    settings["task"]["dim_x"] = train_spectra.shape[1]    
    
    valid_dataset = StandardizedArielDataset(test_targets, test_spectra, preprocessing=train_dataset.preprocessing)
    
    effective_test_spectra = torch.tensor(np.load(join(directory_save, 'effective_test_real_spectra.npy')), dtype=torch.float)
    fake_targets = torch.ones((len(effective_test_spectra), train_targets.shape[1]), dtype=torch.float)
    test_dataset = StandardizedArielDataset(fake_targets, effective_test_spectra, preprocessing=train_dataset.preprocessing)

    return train_dataset, valid_dataset, test_dataset

def load_full_standardized_ariel_dataset(settings):
    directory_save = settings["dataset"]["path"]
    
    ## all training dataset (train + valid) 
    train_targets = torch.tensor(np.load(join(directory_save, 'train_targets.npy')), dtype=torch.float)
    train_spectra =  torch.tensor(np.load(join(directory_save, 'train_ideal_spectra.npy')), dtype=torch.float)
    train_noises = torch.tensor(np.load(join(directory_save, 'train_noises.npy')), dtype=torch.float)
    train_aux_data = torch.tensor(np.load(join(directory_save, 'train_aux_data.npy')), dtype=torch.float)
    
    valid_spectra = torch.tensor(np.load(join(directory_save, 'valid_ideal_spectra.npy')), dtype=torch.float)
    valid_targets = torch.tensor(np.load(join(directory_save, 'valid_targets.npy')), dtype=torch.float)
    valid_noises = torch.tensor(np.load(join(directory_save, 'valid_noises.npy')), dtype=torch.float)
    valid_aux_data = torch.tensor(np.load(join(directory_save, 'valid_aux_data.npy')), dtype=torch.float)
    
    test_spectra = torch.tensor(np.load(join(directory_save, 'test_ideal_spectra.npy')), dtype=torch.float)
    test_noises = torch.tensor(np.load(join(directory_save, 'test_noises.npy')), dtype=torch.float)
    test_aux_data = torch.tensor(np.load(join(directory_save, 'test_aux_data.npy')), dtype=torch.float)
    test_targets = torch.tensor(np.load(join(directory_save, 'test_targets.npy')), dtype=torch.float)

    
    train_dataset = FullStandardizedArielDataset(
        torch.cat([train_targets, valid_targets], dim=0),
        torch.cat([train_spectra, valid_spectra], dim=0),
        torch.cat([train_noises, valid_noises], dim=0),
        torch.cat([train_aux_data, valid_aux_data], dim=0)
    )

    settings["task"]["dim_theta"] = train_targets.shape[1]
    settings["task"]["dim_x"] = train_spectra.shape[1] + train_noises.shape[1] + train_aux_data.shape[1]
    

    valid_dataset = FullStandardizedArielDataset(test_targets, test_spectra, test_noises, test_aux_data, preprocessing=train_dataset.preprocessing)
    
    effective_test_spectra = torch.tensor(np.load(join(directory_save, 'effective_test_ideal_spectra.npy')), dtype=torch.float)
    effective_test_noises = torch.tensor(np.load(join(directory_save, 'effective_test_noises.npy')), dtype=torch.float)
    effective_test_aux_data = torch.tensor(np.load(join(directory_save, 'effective_test_aux_data.npy')), dtype=torch.float)
    fake_targets = torch.ones((len(effective_test_spectra), train_targets.shape[1]), dtype=torch.float)
    test_dataset = FullStandardizedArielDataset(fake_targets, effective_test_spectra, effective_test_noises, effective_test_aux_data, preprocessing=train_dataset.preprocessing)
    
    return train_dataset, valid_dataset, test_dataset

def load_noisy_normalized_ariel_dataset(settings):
    directory_save = settings["dataset"]["path"]
    train_targets = torch.tensor(np.load(join(directory_save, 'train_targets.npy')), dtype=torch.float)
    train_spectra =  torch.tensor(np.load(join(directory_save, 'train_ideal_spectra.npy')), dtype=torch.float)
    train_noises = torch.tensor(np.load(join(directory_save, 'train_noises.npy')), dtype=torch.float)

    valid_spectra = torch.tensor(np.load(join(directory_save, 'valid_ideal_spectra.npy')), dtype=torch.float)
    valid_targets = torch.tensor(np.load(join(directory_save, 'valid_targets.npy')), dtype=torch.float)  
    valid_noises = torch.tensor(np.load(join(directory_save, 'valid_noises.npy')), dtype=torch.float)
    
    test_spectra = torch.tensor(np.load(join(directory_save, 'test_ideal_spectra.npy')), dtype=torch.float)
    test_targets = torch.tensor(np.load(join(directory_save, 'test_targets.npy')), dtype=torch.float)
    test_noises = torch.tensor(np.load(join(directory_save, 'test_noises.npy')), dtype=torch.float)
    
    
    train_dataset = NoisyNormalizedArielDataset(
        torch.cat([train_targets, valid_targets], dim=0),
        torch.cat([train_spectra, valid_spectra], dim=0),
        torch.cat([train_noises, valid_noises], dim=0)
    )

    settings["task"]["dim_theta"] = train_targets.shape[1]
    settings["task"]["dim_x"] = train_spectra.shape[1] + train_noises.shape[1]    
    
    valid_dataset = NoisyNormalizedArielDataset(test_targets, test_spectra, test_noises, preprocessing=train_dataset.preprocessing)
    
    effective_test_spectra = torch.tensor(np.load(join(directory_save, 'effective_test_real_spectra.npy')), dtype=torch.float)
    effective_test_noises = torch.tensor(np.load(join(directory_save, 'effective_test_noises.npy')), dtype=torch.float)
    fake_targets = torch.ones((len(effective_test_spectra), train_targets.shape[1]), dtype=torch.float)
    test_dataset = NoisyNormalizedArielDataset(fake_targets, effective_test_spectra, effective_test_noises, preprocessing=train_dataset.preprocessing)

    return train_dataset, valid_dataset, test_dataset

def load_noisy_standardized_ariel_dataset(settings):
    directory_save = settings["dataset"]["path"]
    train_targets = torch.tensor(np.load(join(directory_save, 'train_targets.npy')), dtype=torch.float)
    train_spectra =  torch.tensor(np.load(join(directory_save, 'train_ideal_spectra.npy')), dtype=torch.float)
    train_noises = torch.tensor(np.load(join(directory_save, 'train_noises.npy')), dtype=torch.float)

    valid_spectra = torch.tensor(np.load(join(directory_save, 'valid_ideal_spectra.npy')), dtype=torch.float)
    valid_targets = torch.tensor(np.load(join(directory_save, 'valid_targets.npy')), dtype=torch.float)  
    valid_noises = torch.tensor(np.load(join(directory_save, 'valid_noises.npy')), dtype=torch.float)
    
    test_spectra = torch.tensor(np.load(join(directory_save, 'test_ideal_spectra.npy')), dtype=torch.float)
    test_targets = torch.tensor(np.load(join(directory_save, 'test_targets.npy')), dtype=torch.float)
    test_noises = torch.tensor(np.load(join(directory_save, 'test_noises.npy')), dtype=torch.float)
    
    
    train_dataset = NoisyStandardizedArielDataset(
        torch.cat([train_targets, valid_targets], dim=0),
        torch.cat([train_spectra, valid_spectra], dim=0),
        torch.cat([train_noises, valid_noises], dim=0)
    )

    settings["task"]["dim_theta"] = train_targets.shape[1]
    settings["task"]["dim_x"] = train_spectra.shape[1] + train_noises.shape[1]    
    
    valid_dataset = NoisyStandardizedArielDataset(test_targets, test_spectra, test_noises, preprocessing=train_dataset.preprocessing)
    
    effective_test_spectra = torch.tensor(np.load(join(directory_save, 'effective_test_real_spectra.npy')), dtype=torch.float)
    effective_test_noises = torch.tensor(np.load(join(directory_save, 'effective_test_noises.npy')), dtype=torch.float)
    fake_targets = torch.ones((len(effective_test_spectra), train_targets.shape[1]), dtype=torch.float)
    test_dataset = NoisyStandardizedArielDataset(fake_targets, effective_test_spectra, effective_test_noises, preprocessing=train_dataset.preprocessing)

    return train_dataset, valid_dataset, test_dataset

def load_real_noisy_normalized_ariel_dataset(settings):
    directory_save = settings["dataset"]["path"]
    train_targets = torch.tensor(np.load(join(directory_save, 'train_targets.npy')), dtype=torch.float)
    train_spectra =  torch.tensor(np.load(join(directory_save, 'train_real_spectra.npy')), dtype=torch.float)
    train_noises = torch.tensor(np.load(join(directory_save, 'train_noises.npy')), dtype=torch.float)

    valid_spectra = torch.tensor(np.load(join(directory_save, 'valid_real_spectra.npy')), dtype=torch.float)
    valid_targets = torch.tensor(np.load(join(directory_save, 'valid_targets.npy')), dtype=torch.float)  
    valid_noises = torch.tensor(np.load(join(directory_save, 'valid_noises.npy')), dtype=torch.float)
    
    test_spectra = torch.tensor(np.load(join(directory_save, 'test_real_spectra.npy')), dtype=torch.float)
    test_targets = torch.tensor(np.load(join(directory_save, 'test_targets.npy')), dtype=torch.float)
    test_noises = torch.tensor(np.load(join(directory_save, 'test_noises.npy')), dtype=torch.float)
    
    
    train_dataset = NoisyNormalizedArielDataset(
        torch.cat([train_targets, valid_targets], dim=0),
        torch.cat([train_spectra, valid_spectra], dim=0),
        torch.cat([train_noises, valid_noises], dim=0)
    )

    settings["task"]["dim_theta"] = train_targets.shape[1]
    settings["task"]["dim_x"] = train_spectra.shape[1] + train_noises.shape[1]    
    
    valid_dataset = NoisyNormalizedArielDataset(test_targets, test_spectra, test_noises, preprocessing=train_dataset.preprocessing)
    
    effective_test_spectra = torch.tensor(np.load(join(directory_save, 'effective_test_real_spectra.npy')), dtype=torch.float)
    effective_test_noises = torch.tensor(np.load(join(directory_save, 'effective_test_noises.npy')), dtype=torch.float)
    effective_test_aux_data = torch.tensor(np.load(join(directory_save, 'effective_test_aux_data.npy')), dtype=torch.float)
    fake_targets = torch.ones((len(effective_test_spectra), train_targets.shape[1]), dtype=torch.float)
    test_dataset = NoisyNormalizedArielDataset(fake_targets, effective_test_spectra, effective_test_noises, effective_test_aux_data, preprocessing=train_dataset.preprocessing)

    return train_dataset, valid_dataset, test_dataset

def load_real_noisy_standardized_ariel_dataset(settings):
    directory_save = settings["dataset"]["path"]
    train_targets = torch.tensor(np.load(join(directory_save, 'train_targets.npy')), dtype=torch.float)
    train_spectra =  torch.tensor(np.load(join(directory_save, 'train_real_spectra.npy')), dtype=torch.float)
    train_noises = torch.tensor(np.load(join(directory_save, 'train_noises.npy')), dtype=torch.float)

    valid_spectra = torch.tensor(np.load(join(directory_save, 'valid_real_spectra.npy')), dtype=torch.float)
    valid_targets = torch.tensor(np.load(join(directory_save, 'valid_targets.npy')), dtype=torch.float)  
    valid_noises = torch.tensor(np.load(join(directory_save, 'valid_noises.npy')), dtype=torch.float)
    
    test_spectra = torch.tensor(np.load(join(directory_save, 'test_real_spectra.npy')), dtype=torch.float)
    test_targets = torch.tensor(np.load(join(directory_save, 'test_targets.npy')), dtype=torch.float)
    test_noises = torch.tensor(np.load(join(directory_save, 'test_noises.npy')), dtype=torch.float)
    
    
    train_dataset = NoisyStandardizedArielDataset(
        torch.cat([train_targets, valid_targets], dim=0),
        torch.cat([train_spectra, valid_spectra], dim=0),
        torch.cat([train_noises, valid_noises], dim=0)
    )

    settings["task"]["dim_theta"] = train_targets.shape[1]
    settings["task"]["dim_x"] = train_spectra.shape[1] + train_noises.shape[1]    
    
    valid_dataset = NoisyStandardizedArielDataset(test_targets, test_spectra, test_noises, preprocessing=train_dataset.preprocessing)
    
    effective_test_spectra = torch.tensor(np.load(join(directory_save, 'effective_test_real_spectra.npy')), dtype=torch.float)
    effective_test_noises = torch.tensor(np.load(join(directory_save, 'effective_test_noises.npy')), dtype=torch.float)
    fake_targets = torch.ones((len(effective_test_spectra), train_targets.shape[1]), dtype=torch.float)
    test_dataset = NoisyStandardizedArielDataset(fake_targets, effective_test_spectra, effective_test_noises, preprocessing=train_dataset.preprocessing)

    return train_dataset, valid_dataset, test_dataset



def load_real_normalized_ariel_dataset(settings):
    directory_save = settings["dataset"]["path"]
    train_targets = torch.tensor(np.load(join(directory_save, 'train_targets.npy')), dtype=torch.float)
    train_spectra =  torch.tensor(np.load(join(directory_save, 'train_real_spectra.npy')), dtype=torch.float)
    train_aux_data = torch.tensor(np.load(join(directory_save, 'train_aux_data.npy')), dtype=torch.float)

    valid_spectra = torch.tensor(np.load(join(directory_save, 'valid_real_spectra.npy')), dtype=torch.float)
    valid_targets = torch.tensor(np.load(join(directory_save, 'valid_targets.npy')), dtype=torch.float)  
    valid_aux_data = torch.tensor(np.load(join(directory_save, 'valid_aux_data.npy')), dtype=torch.float)
    
    test_spectra = torch.tensor(np.load(join(directory_save, 'test_real_spectra.npy')), dtype=torch.float)
    test_targets = torch.tensor(np.load(join(directory_save, 'test_targets.npy')), dtype=torch.float)
    test_aux_data = torch.tensor(np.load(join(directory_save, 'test_aux_data.npy')), dtype=torch.float)
    
    
    train_dataset = RealNormalizedArielDataset(
        torch.cat([train_targets, valid_targets], dim=0),
        torch.cat([train_spectra, valid_spectra], dim=0),
        torch.cat([train_aux_data, valid_aux_data], dim=0)
    )

    settings["task"]["dim_theta"] = train_targets.shape[1]
    settings["task"]["dim_x"] = train_spectra.shape[1] + train_aux_data.shape[1]    
    
    valid_dataset = RealNormalizedArielDataset(test_targets, test_spectra, test_aux_data, preprocessing=train_dataset.preprocessing)
    
    effective_test_spectra = torch.tensor(np.load(join(directory_save, 'effective_test_real_spectra.npy')), dtype=torch.float)
    effective_test_aux_data = torch.tensor(np.load(join(directory_save, 'effective_test_aux_data.npy')), dtype=torch.float)
    fake_targets = torch.ones((len(effective_test_spectra), train_targets.shape[1]), dtype=torch.float)
    test_dataset = RealNormalizedArielDataset(fake_targets, effective_test_spectra, effective_test_aux_data, preprocessing=train_dataset.preprocessing)

    return train_dataset, valid_dataset, test_dataset

def load_real_standardized_ariel_dataset(settings):
    directory_save = settings["dataset"]["path"]
    train_targets = torch.tensor(np.load(join(directory_save, 'train_targets.npy')), dtype=torch.float)
    train_spectra =  torch.tensor(np.load(join(directory_save, 'train_real_spectra.npy')), dtype=torch.float)
    train_aux_data = torch.tensor(np.load(join(directory_save, 'train_aux_data.npy')), dtype=torch.float)

    valid_spectra = torch.tensor(np.load(join(directory_save, 'valid_real_spectra.npy')), dtype=torch.float)
    valid_targets = torch.tensor(np.load(join(directory_save, 'valid_targets.npy')), dtype=torch.float)  
    valid_aux_data = torch.tensor(np.load(join(directory_save, 'valid_aux_data.npy')), dtype=torch.float)
    
    test_spectra = torch.tensor(np.load(join(directory_save, 'test_real_spectra.npy')), dtype=torch.float)
    test_targets = torch.tensor(np.load(join(directory_save, 'test_targets.npy')), dtype=torch.float)
    test_aux_data = torch.tensor(np.load(join(directory_save, 'test_aux_data.npy')), dtype=torch.float)


    train_dataset = RealStandardizedArielDataset(
        torch.cat([train_targets, valid_targets], dim=0),
        torch.cat([train_spectra, valid_spectra], dim=0),
        torch.cat([train_aux_data, valid_aux_data], dim=0)
    )

    settings["task"]["dim_theta"] = train_targets.shape[1]
    settings["task"]["dim_x"] = train_spectra.shape[1] + train_aux_data.shape[1]    
    
    valid_dataset = RealStandardizedArielDataset(test_targets, test_spectra, test_aux_data, preprocessing=train_dataset.preprocessing)
    
        
    effective_test_spectra = torch.tensor(np.load(join(directory_save, 'effective_test_real_spectra.npy')), dtype=torch.float)
    effective_test_aux_data = torch.tensor(np.load(join(directory_save, 'effective_test_aux_data.npy')), dtype=torch.float)
    fake_targets = torch.ones((len(effective_test_spectra), train_targets.shape[1]), dtype=torch.float)
    test_dataset = RealStandardizedArielDataset(fake_targets, effective_test_spectra, effective_test_aux_data, preprocessing=train_dataset.preprocessing)

    return train_dataset, valid_dataset, test_dataset

def load_real_full_normalized_ariel_dataset(settings):
    directory_save = settings["dataset"]["path"]
    train_targets = torch.tensor(np.load(join(directory_save, 'train_targets.npy')), dtype=torch.float)
    train_spectra =  torch.tensor(np.load(join(directory_save, 'train_real_spectra.npy')), dtype=torch.float)
    train_aux_data = torch.tensor(np.load(join(directory_save, 'train_aux_data.npy')), dtype=torch.float)
    train_noises = torch.tensor(np.load(join(directory_save, 'train_noises.npy')), dtype=torch.float)

    valid_spectra = torch.tensor(np.load(join(directory_save, 'valid_real_spectra.npy')), dtype=torch.float)
    valid_targets = torch.tensor(np.load(join(directory_save, 'valid_targets.npy')), dtype=torch.float)  
    valid_aux_data = torch.tensor(np.load(join(directory_save, 'valid_aux_data.npy')), dtype=torch.float)
    valid_noises = torch.tensor(np.load(join(directory_save, 'valid_noises.npy')), dtype=torch.float)
    
    test_spectra = torch.tensor(np.load(join(directory_save, 'test_real_spectra.npy')), dtype=torch.float)
    test_targets = torch.tensor(np.load(join(directory_save, 'test_targets.npy')), dtype=torch.float)
    test_aux_data = torch.tensor(np.load(join(directory_save, 'test_aux_data.npy')), dtype=torch.float)
    test_noises = torch.tensor(np.load(join(directory_save, 'test_noises.npy')), dtype=torch.float)
    
    train_dataset = FullNormalizedArielDataset(
        torch.cat([train_targets, valid_targets], dim=0),
        torch.cat([train_spectra, valid_spectra], dim=0),
        torch.cat([train_noises, valid_noises], dim=0),
        torch.cat([train_aux_data, valid_aux_data], dim=0)
    )
    
    settings["task"]["dim_theta"] = train_targets.shape[1]
    settings["task"]["dim_x"] = train_spectra.shape[1] + train_noises.shape[1] + train_aux_data.shape[1]
    
    valid_dataset = FullNormalizedArielDataset(test_targets, test_spectra, test_noises, test_aux_data, preprocessing=train_dataset.preprocessing)
    
    effective_test_spectra = torch.tensor(np.load(join(directory_save, 'effective_test_real_spectra.npy')), dtype=torch.float)
    effective_test_aux_data = torch.tensor(np.load(join(directory_save, 'effective_test_aux_data.npy')), dtype=torch.float)
    effective_test_noises = torch.tensor(np.load(join(directory_save, 'effective_test_noises.npy')), dtype=torch.float)
    fake_targets = torch.ones((len(effective_test_spectra), train_targets.shape[1]), dtype=torch.float)
    test_dataset = FullNormalizedArielDataset(fake_targets, effective_test_spectra, effective_test_aux_data, preprocessing=train_dataset.preprocessing)

    return train_dataset, valid_dataset, test_dataset
    
def load_real_full_standardized_ariel_dataset(settings):
    directory_save = settings["dataset"]["path"]
    train_targets = torch.tensor(np.load(join(directory_save, 'train_targets.npy')), dtype=torch.float)
    train_spectra =  torch.tensor(np.load(join(directory_save, 'train_real_spectra.npy')), dtype=torch.float)
    train_aux_data = torch.tensor(np.load(join(directory_save, 'train_aux_data.npy')), dtype=torch.float)
    train_noises = torch.tensor(np.load(join(directory_save, 'train_noises.npy')), dtype=torch.float)

    valid_spectra = torch.tensor(np.load(join(directory_save, 'valid_real_spectra.npy')), dtype=torch.float)
    valid_targets = torch.tensor(np.load(join(directory_save, 'valid_targets.npy')), dtype=torch.float)  
    valid_aux_data = torch.tensor(np.load(join(directory_save, 'valid_aux_data.npy')), dtype=torch.float)
    valid_noises = torch.tensor(np.load(join(directory_save, 'valid_noises.npy')), dtype=torch.float)
    
    test_spectra = torch.tensor(np.load(join(directory_save, 'test_real_spectra.npy')), dtype=torch.float)
    test_targets = torch.tensor(np.load(join(directory_save, 'test_targets.npy')), dtype=torch.float)
    test_aux_data = torch.tensor(np.load(join(directory_save, 'test_aux_data.npy')), dtype=torch.float)
    test_noises = torch.tensor(np.load(join(directory_save, 'test_noises.npy')), dtype=torch.float)
    
    train_dataset = FullStandardizedArielDataset(
        torch.cat([train_targets, valid_targets], dim=0),
        torch.cat([train_spectra, valid_spectra], dim=0),
        torch.cat([train_noises, valid_noises], dim=0),
        torch.cat([train_aux_data, valid_aux_data], dim=0)
    )

    settings["task"]["dim_theta"] = train_targets.shape[1]
    settings["task"]["dim_x"] = train_spectra.shape[1] + train_noises.shape[1] + train_aux_data.shape[1]
    
    valid_dataset = FullStandardizedArielDataset(test_targets, test_spectra, test_noises, test_aux_data, preprocessing=train_dataset.preprocessing)
    
    effective_test_spectra = torch.tensor(np.load(join(directory_save, 'effective_test_real_spectra.npy')), dtype=torch.float)
    effective_test_aux_data = torch.tensor(np.load(join(directory_save, 'effective_test_aux_data.npy')), dtype=torch.float)
    effective_test_noises = torch.tensor(np.load(join(directory_save, 'effective_test_noises.npy')), dtype=torch.float)
    fake_targets = torch.ones((len(effective_test_spectra), train_targets.shape[1]), dtype=torch.float)
    test_dataset = FullStandardizedArielDataset(fake_targets, effective_test_spectra, effective_test_noises, effective_test_aux_data, preprocessing=train_dataset.preprocessing)
    
    return train_dataset, valid_dataset, test_dataset
    

def load_dataset(settings):
    # working on real data without noise and aux data
    if settings["dataset"]["type"] == "NormalizedArielDataset":
        return load_normalized_ariel_dataset(settings)
    
    # working on ideal data with noise and aux data
    elif settings["dataset"]["type"] == "FullNormalizedArielDataset":
        return load_full_normalized_ariel_dataset(settings)
    
    # working on real data without noise and aux data
    elif settings["dataset"]["type"] == "StandardizedArielDataset":
        return load_standardized_ariel_dataset(settings)
    
    # working on ideal data with noise and aux data
    elif settings["dataset"]["type"] == "FullStandardizedArielDataset":
        return load_full_standardized_ariel_dataset(settings)
    
    elif settings["dataset"]["type"] == "NoisyNormalizedArielDataset":
        return load_noisy_normalized_ariel_dataset(settings)
    
    elif settings["dataset"]["type"] == "NoisyStandardizedArielDataset":
        return load_noisy_standardized_ariel_dataset(settings)
    
    elif settings["dataset"]["type"] == "RealNoisyNormalizedArielDataset":
        return load_real_noisy_normalized_ariel_dataset(settings)
    
    elif settings["dataset"]["type"] == "RealNoisyStandardizedArielDataset":
        return load_real_noisy_standardized_ariel_dataset(settings)
    
    # working on real data without noise and with aux data
    elif settings["dataset"]["type"] == "RealNormalizedArielDataset":
        return load_real_normalized_ariel_dataset(settings)
    
    # working on real data without noise and with aux data
    elif settings["dataset"]["type"] == "RealStandardizedArielDataset":
        return load_real_standardized_ariel_dataset(settings)
    
    # working on real data with noise and aux data
    elif settings["dataset"]["type"] == "RealFullNormalizedArielDataset":
        return load_real_full_normalized_ariel_dataset(settings)
    
    # working on real data with noise and aux data
    elif settings["dataset"]["type"] == "RealFullStandardizedArielDataset":
        return load_real_full_standardized_ariel_dataset(settings)
    else:
        raise ValueError("Dataset not found")