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


 #%%

def subtract_frames(exposure_sequence): 
    """
    Subtracts frames pairwise: (frame_0 - frame_1), (frame_2 - frame_3), (frame_4 - frame_5), etc.
    This is used for chop subtraction where consecutive pairs represent chopped positions A and B.
    
    Parameters:
    -----------
    exposure_sequence : np.ndarray
        3D array of shape (n_exposures, height, width).
        
    Returns:
    --------
    np.ndarray
        3D array of shape (n_exposures//2, height, width) containing the subtracted frame pairs.
    """
    # Make a new array to hold the subtracted frames
    n_exposures, height, width = exposure_sequence.shape
    subtracted_sequence = np.zeros((n_exposures//2, height, width))

    # Subtract frames pairwise: (0-1), (2-3), (4-5), etc.
    for pair_idx in range(n_exposures // 2):
        frame_a = pair_idx * 2
        frame_b = pair_idx * 2 + 1
        subtracted_sequence[pair_idx] = exposure_sequence[frame_a] - exposure_sequence[frame_b]

    return subtracted_sequence


# Test the frame subtraction
#%%
# Add if name == main block to prevent auto-execution on import
if __name__ == "__main__":
    """ 
    For testing purposes
    """
    # Make exposure sequence for testing 
    from exposure_sequences import make_exposure_sequence
    #%%
    expsoure_sequence = make_exposure_sequence(6, 0.02, self_similar=False)

    # Sequence with correlated drift
    #test_sequence = make_exposure_sequence(6, 0.02, drift={"tau": 0.5, "amp_frac": 0.03}, self_similar=True)

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
    # Show the statistics of the subtracted frames
    for i in range(subtracted_sequence.shape[0]):
        mean = np.mean(subtracted_sequence[i])
        std = np.std(subtracted_sequence[i])
        print(f'Subtracted Frame {i}: Mean = {mean:.2f}, Std = {std:.2f}')
        plt.hist(subtracted_sequence[i].flatten(), bins=50, alpha=0.7)
        plt.show()

# %%
