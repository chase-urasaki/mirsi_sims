#%% 
import sys 
import os 
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt

#%%
from sky_sim import SkyBackgroundExposure
#from sky_sim import SkyBackgroundExposure_with_aperture
# %%
# Generate a sequence of exposures with aperture
from sky_sim import compute_sky_background_rate

#%%
def make_exposure_sequence(n_exposures, exposure_time, self_similar=False): 
    """
    Creates a sequence of sky background limited exposures using an aperture.
    Stores results in 3D array (n_exposures, height, width).
    
    Parameters:
    -----------
    n_exposures : int
        Number of exposures to simulate.
    exposure_time : float
        Exposure time in seconds for each exposure.
    self_similar : bool
        If True, generates the 
    """
    
    # Detector parameters
    read_noise = 800 #e- / read rms
    dark_current = 50 # e- / s / pixel
    array_shape = (240, 320)

    # Initialize the array to hold exposures
    exposures = np.zeros((n_exposures, *array_shape))
    if self_similar:
        # Compute sky background rate 
        sky_background_rate = compute_sky_background_rate(10e-6, 273)

        # Initialize the SkyBackgroundExposure_with_aperture simulator
        simulator = SkyBackgroundExposure(sky_background_rate, read_noise, dark_current)

        # Initialize array to hold exposures
        exposures = np.zeros((n_exposures, *array_shape))

        for i in range(n_exposures):
            frame = simulator.simulate_exposure(exposure_time)
            exposures[i] = frame

        return exposures
    
    else:
        for i in range(n_exposures):
            simulator = SkyBackgroundExposure(compute_sky_background_rate(10e-6, 273), read_noise, dark_current)
            frame = simulator.simulate_exposure(exposure_time)
            exposures[i] = frame

        return exposures
    
#%%
test_sequence = make_exposure_sequence(6, 0.05, self_similar=True)
# %%
for i in range(test_sequence.shape[0]):
    plt.imshow(test_sequence[i], cmap='gray', origin='lower')
    plt.title(f'Exposure {i+1}')
    plt.colorbar(label='Electrons')
    plt.show()
# %%
def coadd_expsoures(expsoure_seuqence, coadds):
    """ 
    This function takes in an exposure sequence and coadds 
    then number of frames by averaging them together.
    
    Parameters:
    -----------
    expsoure_seuqence : np.ndarray
        3D array of exposures (n_exposures, height, width)
    coadds : int
        Number of frames to average together
        
    Returns:
    --------
    coadded_exposures : np.ndarray
        3D array with coadded frames (n_exposures//coadds, height, width)
    """  
    # Initialize array 
    coadded_exposures = np.zeros((len(expsoure_seuqence)//coadds, expsoure_seuqence.shape[1], expsoure_seuqence.shape[2]))
    # Take every cadded frame and average them. e.g. if coadd is two then coadd pairs of image . 
    for i in range(0, len(expsoure_seuqence), coadds):
        coadded_frame = np.mean(expsoure_seuqence[i:i+coadds], axis=0)
        coadded_exposures[i//coadds] = coadded_frame
    # Return a new array with the coadded frames.
    return coadded_exposures

#%%

coadded_sequence = coadd_expsoures(test_sequence, 2)
#%% 
# Show coadded frames 
for i in range(coadded_sequence.shape[0]):
    plt.imshow(coadded_sequence[i], cmap='gray', origin='lower')
    plt.title(f'Coadded Exposure {i+1}')
    plt.colorbar(label='Electrons')
    plt.show()
#%%
#%%
def find_statistics(exposure_sequence):
    """"
    Function takes an exposure sequence (coadded or not), and computes in 5
    """
# %%
