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
    Injects a point source into every chopped frame of the exposure sequence.
    
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
#%%
expsoure_sequence = make_exposure_sequence(6, 0.02, self_similar=False)
 #%%

def subtract_frames(exposure_sequence): 
    """
    Subtracts frames in ABBA sequence static background.
    
    Parameters:
    -----------
    exposure_sequence : np.ndarray
        3D array of shape (n_exposures, height, width).
        
    Returns:
    --------
    np.ndarray
        3D array of shape (n_exposures - 1, height, width) containing the subtracted frames.
    """
    n_exposures = exposure_sequence.shape[0]
    subtracted_sequence = np.zeros((n_exposures - 1, exposure_sequence.shape[1], exposure_sequence.shape[2]))
    
    for i in range(n_exposures - 1):
        subtracted_sequence[i] = exposure_sequence[i + 1] - exposure_sequence[i]
        
    return subtracted_sequence



# Test the frame subtraction
#%%
# Add if name == main block to prevent auto-execution on import
if __name__ == "__main__":
    """ 
    For testing purposes
    """

    # Imshow exposure sequence 
    for i in range(expsoure_sequence.shape[0]):
        plt.imshow(expsoure_sequence[i], cmap='gray', origin='lower')
        plt.title(f'Exposure {i}')
        plt.colorbar(label='Electrons')
        plt.show()

#%%
    # Subtract frames
    subtracted_sequence = subtract_frames(expsoure_sequence)
# %%
    # Show subtracted frames
    for i in range(subtracted_sequence.shape[0]):
        plt.imshow(subtracted_sequence[i], cmap='gray', origin='lower')
        plt.title(f'Subtracted Frame {i+1} - Frame {i}')
        plt.colorbar(label='Electrons')
        plt.show()

# %%
