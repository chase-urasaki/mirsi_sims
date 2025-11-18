#%%
import sys 
import os 
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
#%%
# Inject a point source into an exposure sequence

def mag_to_flux(mag):
    """
    Converts N-band magnitude to flux in Jy."""
    zero_point_flux = 36.0  # Jy for N-band
    flux = zero_point_flux * 10**(-mag / 2.5)
    return flux

def inject_point_source(exposure_sequence, position, mag):
    """
    Injects a point source into each frame of the exposure sequence.
    
    Parameters:
    -----------
    exposure_sequence : np.ndarray
        3D array of shape (n_exposures, height, width).
    position : tuple
        (y, x) coordinates where the point source will be injected.
    mag : float
        Magnitude of the point source to be injected.
    extent: float 
        Radius of the point source to be injected.
        
    Returns:
    --------
    np.ndarray
        Modified exposure sequence with the point source injected.
    """
    n_exposures, height, width = exposure_sequence.shape
    y, x = position
    
    for i in range(n_exposures):
        # Simple injection: add flux to the specified pixel
        exposure_sequence[i, y, x] += mag_to_flux(mag)
        
    return exposure_sequence

# Make exposure sequence for testing 
from exposure_sequences import make_exposure_sequence

expsoure_sequence = make_exposure_sequence(6, 0.02, self_similar=False)
