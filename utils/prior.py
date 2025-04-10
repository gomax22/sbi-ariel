import numpy as np

def default_prior_bounds_level2data():
    """Prior bounds of each different molecules."""

    #### check here!!!!!!####
    T_range = [0,7000]
    gas1_range = [-12, -1]
    gas2_range = [-12, -1]
    gas3_range = [-12, -1]
    gas4_range = [-12, -1]
    gas5_range = [-12, -1]
    
    bounds_matrix = np.vstack([T_range,gas1_range,gas2_range,gas3_range,gas4_range,gas5_range])
    return bounds_matrix

def default_prior_bounds():
    """Prior bounds of each different molecules."""

    #### check here!!!!!!####
    Rp_range = [0.1, 3]
    T_range = [0,7000]
    gas1_range = [-12, -1]
    gas2_range = [-12, -1]
    gas3_range = [-12, -1]
    gas4_range = [-12, -1]
    gas5_range = [-12, -1]
    
    bounds_matrix = np.vstack([Rp_range,T_range,gas1_range,gas2_range,gas3_range,gas4_range,gas5_range])
    return bounds_matrix

def restrict_to_prior(arr, bounds_matrix):
    """Restrict any values within the array to the bounds given by a bounds_matrix.

    Args:
        arr (array): N-D array 
        bounds_matrix (array): an (N, 2) shaped matrix containing the min and max bounds , where N is the number of dimensions

    Returns:
        array: array with extremal values clipped. 
    """
    arr = np.clip(arr, bounds_matrix[:,0],bounds_matrix[:,1])
    return arr