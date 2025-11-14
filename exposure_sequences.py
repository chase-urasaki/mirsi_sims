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
test_sequence = make_exposure_sequence(5, 0.05, self_similar=True)
# %%
for i in range(test_sequence.shape[0]):
    plt.imshow(test_sequence[i], cmap='gray', origin='lower')
    plt.title(f'Exposure {i+1}')
    plt.colorbar(label='Electrons')
    plt.show()
# %%
def coadd_expsoures():  
                    
def find_statistics(exposure_sequence):
    