#%%
import sys 
import os 
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
#%%
# Inject a point source into an exposure sequence
def inject_point_source(exposure_sequence, position, flux):
    """
    Injects a point source into each frame of the exposure sequence.
    
    Parameters:
    -----------
    exposure_sequence : np.ndarray
        3D array of shape (n_exposures, height, width).
    position : tuple
        (y, x) coordinates where the point source will be injected.
    flux : float
        Total flux of the point source to be injected.
        
    Returns:
    --------
    np.ndarray
        Modified exposure sequence with the point source injected.
    """
    n_exposures, height, width = exposure_sequence.shape
    y, x = position
    
    for i in range(n_exposures):
        # Simple injection: add flux to the specified pixel
        exposure_sequence[i, y, x] += flux
        
    return exposure_sequence