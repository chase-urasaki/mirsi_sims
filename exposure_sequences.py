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
# def make_exposure_sequence(n_exposures, exposure_time, drift = None, self_similar=False): 
#     """
#     Creates a sequence of sky background limited exposures using an aperture.
#     Stores results in 3D array (n_exposures, height, width).
    
#     Parameters:
#     -----------
#     n_exposures : int
#         Number of exposures to simulate.
#     exposure_time : float
#         Exposure time in seconds for each exposure.
#     self_similar : bool
#         If True, generates the 
#     """
    
#     # Detector parameters
#     read_noise = 800 #e- / read rms
#     dark_current = 50 # e- / s / pixel
#     array_shape = (240, 320)

#     # Initialize the array to hold exposures
#     exposures = np.zeros((n_exposures, *array_shape))
#     if self_similar:
#         # Compute sky background rate 
#         sky_background_rate = compute_sky_background_rate(10e-6, 273)

#         # Initialize the SkyBackgroundExposure_with_aperture simulator
#         simulator = SkyBackgroundExposure(sky_background_rate, read_noise, dark_current)

#         # Initialize array to hold exposures
#         exposures = np.zeros((n_exposures, *array_shape))

#         for i in range(n_exposures):
#             frame = simulator.simulate_exposure(exposure_time)
            
#             exposures[i] = frame

#         return exposures
    
#     else:
#         for i in range(n_exposures):
#             simulator = SkyBackgroundExposure(compute_sky_background_rate(10e-6, 273), read_noise, dark_current)
#             frame = simulator.simulate_exposure(exposure_time)
#             exposures[i] = frame

#         return exposures
    
#%% 
# test for adding in drift 
import numpy as np

def generate_slow_drift(n_exposures, exposure_time, tau, amp_frac, rng=None):
    """
    Generate a slow, correlated fractional drift term for the sky background.

    Parameters
    ----------
    n_exposures : int
        Number of frames.
    exposure_time : float
        Cadence between frames [s] (assuming back-to-back exposures).
    tau : float
        Correlation timescale of the drift [s].
    amp_frac : float
        RMS amplitude of the fractional drift (e.g. 0.03 for ~3%).
    rng : np.random.Generator, optional
        Random generator for reproducibility.

    Returns
    -------
    drift_frac : (n_exposures,) ndarray
        Zero-mean fractional drift. Use (1 + drift_frac[i]) as the
        multiplicative factor on the *mean* sky background for frame i.
    """
    if rng is None:
        rng = np.random.default_rng()

    dt = exposure_time
    rho = np.exp(-dt / tau)   # correlation between adjacent frames

    x = np.empty(n_exposures, dtype=float)
    x[0] = rng.normal()

    for i in range(1, n_exposures):
        eps = rng.normal()
        x[i] = rho * x[i-1] + np.sqrt(1 - rho**2) * eps

    # normalize to unit variance then scale to desired RMS amplitude
    x /= np.std(x)
    drift_frac = amp_frac * x
    return drift_frac


def make_exposure_sequence(n_exposures, exposure_time, drift=None, self_similar=False): 
    """
    Creates a sequence of sky background limited exposures using an aperture.
    Stores results in 3D array (n_exposures, height, width).
    
    Parameters
    ----------
    n_exposures : int
        Number of exposures to simulate.
    exposure_time : float
        Exposure time in seconds for each exposure.
    drift : None, dict, or array-like
        If None, no slow drift is applied (pure Poisson + detector noise).
        If dict, interpreted as parameters for an OU/AR(1) drift process:
            {
                "tau": <correlation timescale in seconds>,
                "amp_frac": <RMS fractional amplitude, e.g. 0.03>,
                "rng": np.random.Generator (optional)
            }
        If array-like of shape (n_exposures,), it is taken as the
        fractional drift series directly (zero-mean, small amplitude),
        and used as a multiplicative factor: (1 + drift[i]).
    self_similar : bool
        If True, uses a single SkyBackgroundExposure instance and updates its
        sky background rate each frame (preserving any internal structure).
    """
    
    # Detector parameters
    read_noise = 800  # e- / read rms
    dark_current = 50 # e- / s / pixel
    array_shape = (240, 320)

    # Initialize the array to hold exposures
    exposures = np.zeros((n_exposures, *array_shape), dtype=float)

    # --- Base sky background rate (no drift yet) ---
    base_sky_background_rate = compute_sky_background_rate(10e-6, 273)

    # --- Build drift time series if requested ---
    if drift is None:
        drift_series = np.zeros(n_exposures, dtype=float)
    elif isinstance(drift, dict):
        tau = drift["tau"]
        amp_frac = drift["amp_frac"]
        rng = drift.get("rng", None)
        drift_series = generate_slow_drift(
            n_exposures=n_exposures,
            exposure_time=exposure_time,
            tau=tau,
            amp_frac=amp_frac,
            rng=rng
        )
    else:
        # Treat as user-supplied array-like with length n_exposures
        drift_series = np.asarray(drift, dtype=float)
        if drift_series.shape[0] != n_exposures:
            raise ValueError("drift array must have length n_exposures")

    if self_similar:
        # One simulator; we update its sky background rate each frame.
        simulator = SkyBackgroundExposure(
            base_sky_background_rate,
            read_noise,
            dark_current
        )

        for i in range(n_exposures):
            # Apply slow fractional drift to *mean* sky background rate
            sky_background_rate_i = base_sky_background_rate * (1.0 + drift_series[i])
            # Assuming simulator exposes this as an attribute:
            simulator.sky_background_rate = sky_background_rate_i

            frame = simulator.simulate_exposure(exposure_time)
            exposures[i] = frame

    else:
        # New simulator each frame, each with its own drifted background level.
        for i in range(n_exposures):
            sky_background_rate_i = base_sky_background_rate * (1.0 + drift_series[i])

            simulator = SkyBackgroundExposure(
                sky_background_rate_i,
                read_noise,
                dark_current
            )
            frame = simulator.simulate_exposure(exposure_time)
            exposures[i] = frame

    return exposures
#%%
    
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
rng = np.random.default_rng(123)

exposures = make_exposure_sequence(
    n_exposures=60,
    exposure_time=0.02,  # 50 ms
    drift={"tau": 0.5, "amp_frac": 0.03, "rng": rng},
    self_similar=True
)
#%%
plt.imshow(exposures[0], cmap='gray', origin='lower')
plt.colorbar(label='Electrons')
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
