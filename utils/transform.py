import numpy as np

def transform_data(org_arr, aug_arr=None):
    global_mean = np.mean(org_arr)
    global_std = np.std(org_arr)
    if aug_arr is not None:
        std_aug_spectra = standardise(aug_arr, global_mean, global_std)
    else:
        std_aug_spectra = standardise(org_arr, global_mean, global_std)
    std_aug_spectra = std_aug_spectra.reshape(-1, org_arr.shape[1])
    return std_aug_spectra, global_mean,global_std

def standardise(arr, mean, std):
    return (arr-mean)/std

def transform_back(arr, mean, std):
    return arr*std+mean

def transform_and_reshape( y_pred_valid,targets_mean, targets_std,instances,N_testdata):
    y_pred_valid_org = transform_back(y_pred_valid,targets_mean[None, ...], targets_std[None, ...])
    y_pred_valid_org = y_pred_valid_org.reshape(instances, N_testdata, len(targets_std))
    y_pred_valid_org = np.swapaxes(y_pred_valid_org, 1,0)
    return y_pred_valid_org

def normalize(arr):
    min_, max_ = np.min(arr, axis=1), np.max(arr, axis=1)
    return (arr - min_[:, None]) / (max_[:, None] - min_[:, None]), min_, max_

def denormalize(arr, min_, max_):
    return arr*(max_ - min_) + min_

def standardize(arr):
    mean_, std_ = np.mean(arr, axis=1), np.std(arr, axis=1)
    return (arr - mean_[:, None]) / std_[:, None], mean_, std_

def destandardize(arr, mean_, std_):
    return arr*std_ + mean_


